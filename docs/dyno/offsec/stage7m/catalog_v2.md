# CWE reference catalog v2 (Stage 7M revision)

Authoritative source: MITRE Common Weakness Enumeration, cwe.mitre.org
Base catalog: `docs/dyno/offsec/stage7l/catalog.md` (SHA-256
92811f6e0272d0da1c31695f9e289a1baf15e78d94d3b404f14a799d9d97fcf8).

Delta vs the base catalog: EIGHT entries are byte-identical.
TWO entries (CWE-78 and CWE-89) have been rewritten to eliminate the
"constructs all or part of ... command using externally-influenced
input" stem that Stage 7L showed was collapsing them into the same
category under the 2B model's attention pattern. The revised entries
lead with disambiguating context words (database/SQL vs operating
system/shell) and name concrete sink APIs.

Verified: 44 of the 61 CWE-89 errors under Stage 7L condition B (72%)
were CWE-78. See `docs/dyno/offsec/stage7l/cwe89_confusion.json`.

---

CWE-78 (OS Command Injection): The operating-system shell or a subprocess API is invoked with externally-influenced input. Concrete sinks include `os.system`, `subprocess` with `shell=True`, `Runtime.getRuntime().exec("/bin/sh -c ...")`, `exec`/`execSync`/`spawn`/`spawnSync`, backtick or `$(...)` shell templates, and passing concatenated argv into `/bin/sh -c`. The dangerous characters are shell metacharacters (`; & | $() \` <newline>`). SQL databases are not involved; the downstream process is the OS shell.

CWE-79 (Cross-site Scripting): The product does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page served to other users.

CWE-22 (Path Traversal): The product uses external input to construct a pathname that is intended to identify a file or directory located underneath a restricted parent directory, but does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location outside of the restricted directory.

CWE-89 (SQL Injection): A database driver or ORM is passed an SQL query string built from externally-influenced input. Concrete sinks include `cursor.execute`, `db.query`, `knex.raw`, `sa.text`, `PDO::prepare` or `PDO::query` with concatenation, `gorm.Raw`, `Jdbc.executeQuery`, and any ORM `raw` / `text` / `execSQL` method whose argument is a concatenated or f-string SQL string. The dangerous characters are SQL metacharacters (single quote `'`, double-dash comment `--`, semicolon `;`, UNION, OR 1=1). Operating-system shells are not involved; the downstream processor is a database.

CWE-311 (Missing Encryption of Sensitive Data): The product does not encrypt sensitive or critical information before storage or transmission. Examples include storing passwords in plaintext, logging credit-card data, or omitting the Secure attribute on session cookies.

CWE-327 (Use of a Broken or Risky Cryptographic Algorithm): The product uses a broken or risky cryptographic algorithm or protocol. Examples include MD5 or SHA-1 for password hashing, DES or RC4 for encryption, or hard-coded symmetric keys.

CWE-346 (Origin Validation Error): The product does not properly verify that the source of data or communication is valid. Examples include trusting the Origin, Referer, or X-Forwarded-For HTTP header as an authentication or authorization decision.

CWE-347 (Improper Verification of Cryptographic Signature): The product does not verify, or incorrectly verifies, the cryptographic signature of data. Examples include JWT alg=none acceptance, HS256/RS256 algorithm confusion, calls that disable signature verification, or ignoring the error returned by a verification function.

CWE-352 (Cross-Site Request Forgery, CSRF): The web application does not, or cannot, sufficiently verify whether a well-formed, valid, consistent request was intentionally provided by the user who submitted it. Examples include state-changing endpoints with no CSRF token, state-changing GET endpoints, or JSON APIs without Origin checks.

CWE-434 (Unrestricted Upload of File with Dangerous Type): The product allows the upload of a file to a location where it can be executed or interpreted (for example the webroot), without properly validating its type or sanitizing its name. Examples include servlets that store uploads under the client-supplied filename, handlers that trust the Content-Type request header, or extraction code that writes a file wherever the archive asked for.
