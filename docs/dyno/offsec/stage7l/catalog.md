# CWE reference catalog (for condition B)

Authoritative source: MITRE Common Weakness Enumeration, cwe.mitre.org
Catalog version used: CWE 4.14 (as embedded in training data; MITRE
publishes under CC BY 4.0). All ten identifiers cover the full
Stage 7K / Stage 7L evaluation set. Per the Stage 7L brief we do NOT
assume the published CWE Top-25 contains every required category;
specifically CWE-346 is NOT in the Top-25, and CWE-311 moves in and
out of the Top-25 across years — both are included below regardless.

The text below (and nothing else from this document) is what gets
spliced into condition B's prompt. No marketing framing, no
examples — just the identifier, name, and one-sentence functional
description from the MITRE abstraction level.

---

CWE-78 (OS Command Injection): The product constructs all or part of an OS command using externally-influenced input from an upstream component, but does not neutralize or incorrectly neutralizes special elements that could modify the intended OS command when it is sent to a downstream component.

CWE-79 (Cross-site Scripting): The product does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page served to other users.

CWE-22 (Path Traversal): The product uses external input to construct a pathname that is intended to identify a file or directory located underneath a restricted parent directory, but does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location outside of the restricted directory.

CWE-89 (SQL Injection): The product constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component.

CWE-311 (Missing Encryption of Sensitive Data): The product does not encrypt sensitive or critical information before storage or transmission. Examples include storing passwords in plaintext, logging credit-card data, or omitting the Secure attribute on session cookies.

CWE-327 (Use of a Broken or Risky Cryptographic Algorithm): The product uses a broken or risky cryptographic algorithm or protocol. Examples include MD5 or SHA-1 for password hashing, DES or RC4 for encryption, or hard-coded symmetric keys.

CWE-346 (Origin Validation Error): The product does not properly verify that the source of data or communication is valid. Examples include trusting the Origin, Referer, or X-Forwarded-For HTTP header as an authentication or authorization decision.

CWE-347 (Improper Verification of Cryptographic Signature): The product does not verify, or incorrectly verifies, the cryptographic signature of data. Examples include JWT alg=none acceptance, HS256/RS256 algorithm confusion, calls that disable signature verification, or ignoring the error returned by a verification function.

CWE-352 (Cross-Site Request Forgery, CSRF): The web application does not, or cannot, sufficiently verify whether a well-formed, valid, consistent request was intentionally provided by the user who submitted it. Examples include state-changing endpoints with no CSRF token, state-changing GET endpoints, or JSON APIs without Origin checks.

CWE-434 (Unrestricted Upload of File with Dangerous Type): The product allows the upload of a file to a location where it can be executed or interpreted (for example the webroot), without properly validating its type or sanitizing its name. Examples include servlets that store uploads under the client-supplied filename, handlers that trust the Content-Type request header, or extraction code that writes a file wherever the archive asked for.
