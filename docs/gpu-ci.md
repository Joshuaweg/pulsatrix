# GPU CI (self-hosted runner)

GitHub's hosted runners have no AMD GPU, so the HIP backend's tests run on a machine with one: a
self-hosted runner on the maintainer's gfx1151 (Strix Halo). The workflow is
`.github/workflows/gpu.yml`. On every push to this repository it runs two jobs on that machine:

| Job | What it runs |
|---|---|
| `hip-tests` | The host kernel check, a Debug build in the ROCm 10.0.0 container, the full test suite, then the GPU tests again with a wait after every op (`PULSATRIX_HIP_SYNC_DEBUG=1`), so a faulting kernel is reported at the op that caused it |
| `hip-bench` | A Release build of `pulsatrix_bench`, the benchmarks on the GPU with the LRP conservation gate (`--check`), and the report as an artifact |

Builds are incremental: the runner keeps its build directories between runs. A pull request
from a branch of this repository shows the results on its commits. Pull requests from forks
don't run on the GPU; see below.

## Security

The repository is public, and a self-hosted runner executes whatever a workflow tells it to, on
the machine it runs on. Three things keep other people's code off it:

1. **The workflow runs only on pushes** to this repository's branches, which only collaborators
   can make, and by hand (`workflow_dispatch`). It never runs on `pull_request` or
   `pull_request_target`, which a fork can trigger. Its jobs also check that they run in this
   repository, not in a fork that copied the workflow.
2. **Workflows from forks need approval.** A fork's pull request can add a workflow of its own
   that asks for this runner. In **Settings → Actions → General → Approval for running fork pull
   request workflows from contributors**, choose **Require approval for all external
   contributors**, and don't approve a run you haven't read.
3. **The runner is a dedicated user** (`pulsatrix-runner`) with no sudo. The tests run inside the
   ROCm container. The user is in the `docker` group, which the container needs and which is
   close to root access on the machine. That is why points 1 and 2 matter.

## Setting up the runner

Once, on the GPU machine, from a checkout of this repository:

1. Check the repository setting in point 2 above.
2. Get a registration token: **Settings → Actions → Runners → New self-hosted runner**, Linux,
   x64. The token is the value after `--token` on that page, and is valid for one hour.
3. Run:

    ```bash
    sudo scripts/setup_gpu_runner.sh TOKEN
    ```

The script checks the host kernel (`scripts/check_host_kernel.sh`), Docker and the ROCm image.
It then creates the `pulsatrix-runner` user and downloads the GitHub Actions runner, at a pinned
version with its checksum verified. Finally it registers the runner with the `gfx1151` label and
installs it as a systemd service, so it starts with the machine. The runner then shows as
**Idle** under **Settings → Actions → Runners**, and the next push runs on it.

If the `pulsatrix-rocm:10.0.0` image hasn't been built on the machine, the first run builds it
(about 20 minutes). Running the script again re-registers the runner.

### Optional: the published ResNet18 and VGG16

The vision tests also check the published torchvision weights when `PULSATRIX_TORCHVISION_DIR`
names a directory holding them (see the
[ImageNet recipe](recipes/interpretability/imagenet_lrp.md#checked-against-zennit)). The tests run
in a container that sees only the checkout, so the directory must be inside the runner's checkout
(`build-ci-hip` survives between runs), given relative to it:

```bash
sudo -u pulsatrix-runner mkdir -p ~pulsatrix-runner/actions-runner/_work/pulsatrix/pulsatrix/build-ci-hip/tv
# copy resnet18.safetensors, vgg16.safetensors and their *_golden.safetensors there, then:
echo 'PULSATRIX_TORCHVISION_DIR=build-ci-hip/tv' | sudo -u pulsatrix-runner tee -a ~pulsatrix-runner/actions-runner/.env
sudo systemctl restart 'actions.runner.*'
```

## Removing the runner

Remove it under **Settings → Actions → Runners**, then on the machine:

```bash
cd ~pulsatrix-runner/actions-runner && sudo ./svc.sh stop && sudo ./svc.sh uninstall
sudo userdel -r pulsatrix-runner
```
