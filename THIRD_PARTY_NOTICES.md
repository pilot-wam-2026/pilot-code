# Third-Party Notices

Removing first-party personal paths and optional bylines does not remove
legally required third-party attribution. Upstream authors are not being
presented as authors of the anonymous submission.

| Component | Retained notice or scope |
|---|---|
| StarVLA-derived model/training framework | Existing source headers and `LICENSES/StarVLA-MIT.txt` |
| OpenVLA-derived logging utilities | `LICENSES/OpenVLA-MIT.txt` |
| NVIDIA GR00T-derived data and simulation utilities | Existing Apache-2.0 headers |
| VJEPA2 source | `starVLA/facebookresearch_vjepa2_main/LICENSE` and `APACHE-LICENSE` |
| NVIDIA Cosmos pretrained components | `NOTICE` and `LICENSES/NVIDIA-Open-Model-License.html`; no relicensing of weights |
| VJEPA2-AC pretrained components | Original model terms remain applicable; source MIT license is not a blanket weight license |
| RoboCasa | `third_party/robocasa/LICENSE` |
| RoboSuite | `third_party/robosuite/LICENSE` |
| hrl_geom and pykdl_utils | Original source headers and package manifests, including BSD notices |
| Orocos KDL / PyKDL | `third_party/orocos_kinematics_dynamics/orocos_kdl/COPYING` and source notices |
| pybind11 | Vendored `LICENSE` within the KDL source tree |
| GR1 geometry and RoboCasa/RoboSuite objects | Embedded asset notices and applicable original asset terms |
| Tokenizer/model metadata | Original upstream component terms |

The Hugging Face resource repository's existing visibility is preserved.
Making it public requires confirming the applicable model and asset
redistribution terms; this package does not infer permission solely from
files being present on a research server.

The model/tokenizer metadata and complete checkpoint are used together;
downloading an unrelated newer backbone is not an equivalent reconstruction.
Third-party code is retained only as required by this model's training and
evaluation stack, including ancestor classes and shared dependency helpers.

## License Sources

The standalone texts were retrieved from their upstream publishers on
September 27, 2026:

- [StarVLA MIT license](https://github.com/starVLA/starVLA/blob/starVLA_dev/LICENSE)
- [OpenVLA MIT license](https://github.com/openvla/openvla/blob/main/LICENSE)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

The NVIDIA agreement retained here states a version release date of
October 24, 2025. The copy preserves the complete agreement text, including
its separate-component, attribution, guardrail, and trustworthy-AI terms.
`LICENSES/Apache-2.0.txt` is also retained for Apache-licensed source files.

## Deployment Boundary

The inherited Cosmos wrapper uses an optional no-op safety-checker adapter
for the archived robotics research path. This release does not establish
that this adapter constitutes a compliant replacement guardrail for any
particular use or deployment. Review the applicable upstream agreement
and provide appropriate safeguards before distributing a service or
using the model outside the controlled simulation benchmark. Do not
interpret private repository access as a waiver of upstream terms.
