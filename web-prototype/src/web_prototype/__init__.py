"""Web prototype for the Mastercard Innovation Challenge submission.

Read-only UI layer over identify/, generate/, defend/, and mandate-demo/.
Nothing in those four directories is modified or reimplemented here --
this package only reads their files and imports mandate-demo's actual
code (installed as a local editable dependency) to run real checks.
"""
