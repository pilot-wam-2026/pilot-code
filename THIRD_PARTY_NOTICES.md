# Third-Party Sources

## Attribution And Anonymity

Copyright notices, license text, dependency-author metadata, and upstream
links are retained as third-party attribution. They are not an author list
or affiliation statement for this submission. Optional personal bylines,
personal filesystem paths, and historical experiment records are not part
of the anonymous source release.

## Dependency Scope

This directory is a private reproducibility archive, not a declaration that
all upstream code, weights, and assets have one common license.

- RoboCasa: original license is retained at `third_party/robocasa/LICENSE`.
- RoboSuite: upstream license is retained at `third_party/robosuite/LICENSE`.
  Its collection source is recorded in
  `provenance/runtime_versions_and_licenses.json`.
- pykdl_utils and hrl_geom: original `package.xml` and source copyright
  headers are retained; both package manifests declare BSD. No standalone
  LICENSE file was found in the copied upstream roots. Resolve the complete
  applicable license text before public redistribution.
- Orocos KDL / PyKDL 1.5.4: the recovered source and its original notices,
  including `orocos_kdl/COPYING`, are vendored. Its pybind11 source/notices
  are also retained. See `provenance/kdl_source.json` for the source revision.
- GR1 robot geometry and RoboCasa object assets: preserve all embedded
  notices; their redistribution rights must be reviewed separately.
- Original starVLA, NVIDIA simulation wrappers, VJEPA, model metadata, and
  other source headers/notices are preserved. Backbone weight licenses must
  also be reviewed before distribution; do not relicense them as project code.

The future Hugging Face upload is on hold. No public-distribution license or
permission has been inferred from local access to these resources.
