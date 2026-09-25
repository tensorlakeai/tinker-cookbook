"""
Thin wrapper around the Tensorlake Sandbox API.

Tensorlake provides cloud microVM sandboxes that start in well under a second,
support snapshot/restore, and can expose ports inside the sandbox to the client.
Requires a Tensorlake API key: set ``TENSORLAKE_API_KEY``.

Configuration via environment variables:
    TENSORLAKE_API_KEY: API key (required)
    TENSORLAKE_API_URL: API endpoint (default: https://api.tensorlake.ai)
    TENSORLAKE_MAX_CONCURRENCY: Max concurrent sandboxes in TensorlakeSandboxPool (default: 32)

See: https://docs.tensorlake.ai/sandboxes/introduction
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path

try:
    from tensorlake.sandbox import AsyncSandbox, AsyncTcpTunnel, SandboxNotFoundError
except ImportError:
    raise ImportError(
        "tensorlake is required for TensorlakeSandbox. "
        "Install it with: uv pip install 'tinker-cookbook[tensorlake] @ "
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'"
    ) from None

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024
_TERMINATED_RE = re.compile(
    r"sandbox '[^']*' (not found|not running|terminated)|sandbox (has been )?terminated", re.I
)


def _is_sandbox_terminated(e: BaseException) -> bool:
    """Check if an exception indicates the sandbox has died."""
    if isinstance(e, SandboxNotFoundError):
        return True
    # A dead sandbox gives a 404 "Sandbox '<id>' not found or not running".
    # A missing file also gives a 404, so match on the message.
    return _TERMINATED_RE.search(str(e)) is not None


# AsyncSandbox.run() buffers all output on the client before it returns, so cap
# stdout/stderr inside the sandbox. Each stream goes through ``head -c N``, then
# ``cat`` discards the rest so the program does not get SIGPIPE. ``stdbuf -o0``
# stops ``head`` from buffering, so output before a timeout kill is kept.
# $1 is the command and $2 is the cap. ``wait`` on process-substitution PIDs
# needs bash 4.4+.
_CAPPED_RUN_SCRIPT = """\
n=$2
cap() {
  if command -v stdbuf >/dev/null 2>&1; then stdbuf -o0 head -c "$n"; else head -c "$n"; fi
  cat >/dev/null
}
exec 5> >(cap); p1=$!
exec 6> >(cap >&2); p2=$!
bash -lc "$1" >&5 2>&6; rc=$?
exec 5>&- 6>&-
wait "$p1" "$p2"
exit "$rc"
"""


def _cap(text: str, max_bytes: int) -> str:
    """Truncate *text* to at most *max_bytes* UTF-8 bytes."""
    data = text.encode()
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")


class TensorlakeSandbox:
    """
    Persistent Tensorlake sandbox for code execution. Conforms to SandboxInterface.

    Usage:
        sandbox = await TensorlakeSandbox.create()

        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python3 /workspace/code.py")
        print(result.stdout)

        await sandbox.cleanup()

    Beyond SandboxInterface, this class supports long-running services
    (``start_process`` + ``open_port``) and snapshots (``checkpoint`` +
    ``create(snapshot_id=...)``), which agent environments such as MCP
    gateways need.
    """

    def __init__(
        self,
        sandbox: AsyncSandbox,
        max_stream_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        user: str | None = "root",
    ) -> None:
        self._sandbox = sandbox
        self._user = user
        self._max_stream_output_bytes = max_stream_output_bytes
        self._tunnels: dict[int, AsyncTcpTunnel] = {}
        self._cleaned_up = False

    @classmethod
    async def create(
        cls,
        image: str | None = None,
        timeout: int = 600,
        cpus: float | None = None,
        memory_mb: int | None = None,
        snapshot_id: str | None = None,
        allow_internet_access: bool = True,
        allow_out: list[str] | None = None,
        max_stream_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        user: str | None = "root",
    ) -> TensorlakeSandbox:
        """Create a new Tensorlake sandbox.

        Args:
            image: Registered sandbox image name. None uses the Tensorlake default
                image (Ubuntu with Python 3). See ``build_image_from_dockerfile``.
            timeout: Lifetime of the sandbox in seconds.
            cpus: CPUs for the sandbox. None uses the server default.
            memory_mb: Memory for the sandbox in MB. None uses the server default.
            snapshot_id: Restore from this snapshot (see ``checkpoint``) instead
                of booting a fresh image.
            allow_internet_access: If False, block all outbound network traffic.
            allow_out: If set, allow outbound traffic only to these destinations.
            max_stream_output_bytes: Default cap for stdout/stderr per command.
            user: User that runs commands. Defaults to root, like Modal. None
                uses the image's ``USER`` (the default image uses a non-root
                user). File reads and writes that the image's user cannot do
                fall back to this user.
        """
        sandbox = await AsyncSandbox.create(
            image=image,
            timeout_secs=timeout,
            cpus=cpus,
            memory_mb=memory_mb,
            snapshot_id=snapshot_id,
            allow_internet_access=allow_internet_access,
            allow_out=allow_out,
        )
        return cls(sandbox=sandbox, max_stream_output_bytes=max_stream_output_bytes, user=user)

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.sandbox_id

    async def send_heartbeat(self, timeout: int = 30) -> None:
        try:
            await asyncio.wait_for(self._sandbox.run("true"), timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            raise

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        On timeout, the process is killed and ``exit_code`` is -9.
        """
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        try:
            result = await self._sandbox.run(
                "bash",
                ["-c", _CAPPED_RUN_SCRIPT, "bash", command, str(cap)],
                working_dir=workdir,
                timeout=timeout,
                user=self._user,
            )
            return SandboxResult(
                stdout=_cap(result.stdout, cap),
                stderr=_cap(result.stderr, cap),
                exit_code=result.exit_code,
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox."""
        try:
            data = await asyncio.wait_for(self._sandbox.read_file(path), timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            # The file API runs as the image's default user, which may not be
            # able to read the file. Read it as ``self._user`` instead.
            if max_bytes is not None:
                cmd = f"head -c {max_bytes} {shlex.quote(path)}"
            else:
                cmd = f"cat {shlex.quote(path)}"
            return await self.run_command(cmd, timeout=timeout)
        content: bytes = data.value
        if max_bytes is not None:
            content = content[:max_bytes]
        return SandboxResult(
            stdout=content.decode("utf-8", errors="replace"), stderr="", exit_code=0
        )

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox. Parent directories are created."""
        if isinstance(content, str):
            content = content.encode()
        try:
            await asyncio.wait_for(self._sandbox.write_file(path, content), timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            # The file API runs as the image's default user, which may not be
            # able to write to *path*. Stage the file in /tmp and move it
            # into place as ``self._user``.
            staged = f"/tmp/.tinker-upload-{uuid.uuid4().hex}"
            try:
                await asyncio.wait_for(self._sandbox.write_file(staged, content), timeout=timeout)
            except Exception as e2:
                if _is_sandbox_terminated(e2):
                    raise SandboxTerminatedError(str(e2)) from e2
                return SandboxResult(stdout="", stderr=str(e2), exit_code=-1)
            quoted = shlex.quote(path)
            result = await self.run_command(
                f"mkdir -p {shlex.quote(os.path.dirname(path) or '/')} && mv {staged} {quoted}",
                timeout=timeout,
            )
            if result.exit_code != 0:
                return result
        if executable:
            return await self.run_command(f"chmod +x {shlex.quote(path)}", timeout=timeout)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def start_process(
        self,
        command: str,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Start a long-running shell command in the background. Returns its PID.

        Use this for services inside the sandbox (for example an MCP gateway),
        then reach them with ``open_port``.
        """
        try:
            proc = await self._sandbox.start_process(
                "bash", ["-lc", command], env=env, working_dir=workdir, user=self._user
            )
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            raise SandboxError(f"Failed to start process: {e}") from e
        return proc.pid

    async def open_port(self, port: int) -> str:
        """Return a local base URL (``http://127.0.0.1:<n>``) that reaches *port* in the sandbox.

        The connection goes through an authenticated tunnel, so the port is not
        exposed to the internet. Tunnels are closed by ``cleanup``.
        """
        tunnel = self._tunnels.get(port)
        if tunnel is None or tunnel.closed:
            tunnel = await self._sandbox.create_tunnel(port, local_port=0)
            self._tunnels[port] = tunnel
        return f"http://{tunnel.local_host}:{tunnel.local_port}"

    async def checkpoint(self, timeout: float = 300) -> str:
        """Snapshot the sandbox state and return the snapshot ID.

        Pass the ID to ``TensorlakeSandbox.create(snapshot_id=...)`` to start
        new sandboxes from this state (for example, one per rollout in a group).
        """
        snapshot = await self._sandbox.checkpoint(timeout=timeout)
        if snapshot is None:
            raise SandboxError("Tensorlake checkpoint returned no snapshot")
        return snapshot.snapshot_id

    async def cleanup(self) -> None:
        """Close tunnels and terminate the sandbox. Safe to call multiple times."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        for tunnel in self._tunnels.values():
            with contextlib.suppress(Exception):
                await tunnel.close()
        self._tunnels.clear()
        try:
            await self._sandbox.terminate()
        except Exception as e:
            if not _is_sandbox_terminated(e):
                raise


class TensorlakeSandboxPool:
    """
    Concurrency-limited executor for one-shot runs (for example, grading code).

    Tensorlake sandboxes start in under a second, so the pool does not keep
    warm sandboxes. Each call creates a fresh sandbox, runs the command, and
    terminates it. At most ``max_concurrency`` sandboxes run at once.

    Has the same ``run_in_workdir`` / ``terminate`` API as ModalSandboxPool.

    Configuration via environment variables:
        TENSORLAKE_MAX_CONCURRENCY: Max concurrent sandboxes (default: 32)
    """

    def __init__(
        self,
        *,
        max_concurrency: int | None = None,
        sandbox_timeout_secs: int = 1200,
        image: str | None = None,
        setup_command: str | None = None,
    ):
        """
        Args:
            max_concurrency: Max sandboxes that run at the same time.
            sandbox_timeout_secs: Lifetime of each sandbox in seconds.
            image: Registered sandbox image name. None uses the default image.
            setup_command: Optional shell command that runs once in a template
                sandbox (for example ``pip install numpy``). The result is
                snapshotted and every run starts from that snapshot.
        """
        self._max_concurrency = max_concurrency or int(
            os.getenv("TENSORLAKE_MAX_CONCURRENCY", "32")
        )
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._image = image
        self._setup_command = setup_command
        self._snapshot_id: str | None = None
        self._setup_lock = asyncio.Lock()
        self._active: set[TensorlakeSandbox] = set()
        self._creating: set[asyncio.Task[TensorlakeSandbox]] = set()
        self._terminated = False

    def _check_terminated(self) -> None:
        if self._terminated:
            raise SandboxError("TensorlakeSandboxPool has been terminated.")

    async def _create_sandbox(self, snapshot_id: str | None = None) -> TensorlakeSandbox:
        """Create a sandbox and add it to ``_active``.

        ``terminate`` waits for creations in progress, so no sandbox is left
        running after it returns.
        """
        self._check_terminated()
        task = asyncio.ensure_future(
            TensorlakeSandbox.create(
                image=self._image, timeout=self._sandbox_timeout_secs, snapshot_id=snapshot_id
            )
        )
        self._creating.add(task)
        try:
            sandbox = await task
        finally:
            self._creating.discard(task)
        if self._terminated:
            await sandbox.cleanup()
            self._check_terminated()
        self._active.add(sandbox)
        return sandbox

    async def _release(self, sandbox: TensorlakeSandbox) -> None:
        self._active.discard(sandbox)
        try:
            await sandbox.cleanup()
        except Exception as e:
            logger.warning(f"Tensorlake sandbox cleanup failed: {e}")

    async def _get_snapshot_id(self) -> str | None:
        """Run ``setup_command`` once and snapshot the result."""
        if self._setup_command is None:
            return None
        async with self._setup_lock:
            if self._snapshot_id is None:
                template = await self._create_sandbox()
                try:
                    result = await template.run_command(self._setup_command, timeout=600)
                    self._check_terminated()
                    if result.exit_code != 0:
                        raise SandboxError(
                            f"Tensorlake pool setup failed ({result.exit_code}): {result.stderr}"
                        )
                    self._snapshot_id = await template.checkpoint()
                finally:
                    await self._release(template)
            return self._snapshot_id

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """
        Execute command with files in a fresh sandbox.
        If ``max_concurrency`` sandboxes are busy, waits until one finishes.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds
        """
        self._check_terminated()
        snapshot_id = await self._get_snapshot_id()
        async with self._semaphore:
            sandbox = await self._create_sandbox(snapshot_id)
            try:
                workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
                if files:
                    results = await asyncio.gather(
                        *(
                            sandbox.write_file(f"{workdir}/{filename}", content)
                            for filename, content in files.items()
                        )
                    )
                    for r in results:
                        if r.exit_code != 0:
                            return r
                else:
                    await sandbox.run_command(f"mkdir -p {shlex.quote(workdir)}")
                return await sandbox.run_command(
                    shlex.join(command),
                    workdir=workdir,
                    timeout=timeout or self._sandbox_timeout_secs,
                )
            finally:
                await self._release(sandbox)

    async def terminate(self) -> None:
        """Stop accepting work and terminate all sandboxes, including ones being created."""
        self._terminated = True
        # Each creation cleans up its own sandbox when it sees ``_terminated``.
        await asyncio.gather(*list(self._creating), return_exceptions=True)
        active, self._active = list(self._active), set()
        await asyncio.gather(*(sb.cleanup() for sb in active), return_exceptions=True)


def build_image_from_dockerfile(
    dockerfile_path: str | Path,
    context_dir: str | Path | None = None,
    name: str | None = None,
    rebuild: bool = False,
) -> str:
    """Build a Tensorlake sandbox image from a Dockerfile and return its registered name.

    The image is cached by name. The default name comes from a hash of the
    Dockerfile content and the context path, so a second call with the same
    inputs does not rebuild. Pass ``rebuild=True`` after you change files in
    the context directory.

    This call blocks while the image builds. In async code, run it with
    ``asyncio.to_thread``.

    Args:
        dockerfile_path: Path to the Dockerfile.
        context_dir: Build context directory. Defaults to the Dockerfile's directory.
        name: Registered image name. Defaults to ``tinker-<hash>``.
        rebuild: Build even if an image with this name exists.
    """
    from tensorlake.image.sandbox_builder import build_sandbox_image, find_sandbox_image_by_name

    dockerfile = Path(dockerfile_path).resolve()
    context = Path(context_dir).resolve() if context_dir is not None else dockerfile.parent
    if name is None:
        digest = hashlib.sha256(dockerfile.read_bytes() + str(context).encode()).hexdigest()
        name = f"tinker-{digest[:16]}"

    if not rebuild and find_sandbox_image_by_name(name) is not None:
        return name

    # Tensorlake uses the Dockerfile's directory as the build context, so put
    # a copy of the Dockerfile in the context directory when they differ.
    if context == dockerfile.parent:
        build_sandbox_image(str(dockerfile), registered_name=name)
    else:
        tmp_dockerfile = context / f".tinker-{uuid.uuid4().hex[:8]}.Dockerfile"
        shutil.copyfile(dockerfile, tmp_dockerfile)
        try:
            build_sandbox_image(str(tmp_dockerfile), registered_name=name)
        finally:
            tmp_dockerfile.unlink(missing_ok=True)
    return name
