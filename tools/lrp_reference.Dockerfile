# Reproduces the pinned environment tools/generate_lrp_reference_values.py was run in to produce
# the hardcoded reference values in tests/lrp_reference_test.cpp (Zennit / LXT). Not used by the
# build or CI.
#
#   docker build -t pulsatrix-lrpref:latest -f tools/lrp_reference.Dockerfile tools
#   docker run --rm -v "$PWD":/w -w /w pulsatrix-lrpref:latest python3 tools/generate_lrp_reference_values.py
#
# The final --force-reinstall --no-deps puts the CPU torchvision back: lxt's dependency resolution
# otherwise replaces it with a build that does not match the CPU torch wheel. transformers is
# pinned because lxt 2.1 depends on it (the generator itself only uses lxt.explicit).
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir zennit==1.0.0 lxt==2.1 numpy==2.5.2 transformers==5.18.0 \
    && pip install --no-cache-dir --force-reinstall --no-deps torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu
