"""Thin Python connectors — the ones whose only stable integration point is
an official Python SDK rather than a Rust crate,
plus the LanceDB fallback used by builds that leave the native LanceDB
connector out.

Every connector here follows the same design constraints as the native Rust
ones in `vechealth-connectors`: no ANN/search path is ever used to bulk-read
data, no sampling or row cap is applied, and errors are raised as
`vechealth.ConnectorError` so callers don't need to know whether a given
connector happens to be Rust or Python underneath.
"""
