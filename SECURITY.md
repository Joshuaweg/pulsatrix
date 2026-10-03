# Security Policy

Pulsatrix is a research/educational C++ deep learning library. It is not
designed to process untrusted input (model files, tensors, or configuration)
from an adversarial source — treat it the same as any other numerical
library you link into your own application's trust boundary.

## Reporting a Vulnerability

If you find a security issue (memory-safety bug, unsafe deserialization path,
etc.), please report it privately rather than opening a public issue:

- Preferred: open a private [GitHub Security Advisory](https://github.com/Joshuaweg/pulsatrix/security/advisories/new)
  on this repository.
- Alternative: email the maintainer directly (see the GitHub profile for
  contact info).

This is a solo-maintained project, so there's no formal SLA, but reports will
be acknowledged and triaged as soon as possible.
