# R7 one-shot confirmatory execution audit

* run id: `r7-1789631512-34220`
* first touch: **2026-09-17T07:51:56Z**
* `HOLDOUT_TOUCHED = YES`
* ledger: `out\r7_confirmatory_kernel_ranking_20260917\confirmatory\CONFIRMATORY_TOUCH_ONCE__411_420.json` (O_CREAT|O_EXCL, created before the first holdout byte)
* pre-execution verification: **PASS**
* confirmatory seed block: `[411, 412, 413, 414, 415, 416, 417, 418, 419, 420]`
* units written: **30** of 30
* unit failures: **0**
* Gate D: **PASS**

## Irreversibility

The ledger file exists. Every future invocation of the confirmatory executor refuses to run, regardless of outcome. A crash does not grant a retry: the run would be recorded as `INCOMPLETE_FIRST_CONFIRMATORY_EXECUTION`.

## Locked hashes

* locked spec sha256: `9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d`
* executor sha256: `5a125184154213c736a41bff8597237d01444372c0b805f35cb6ce4ca552f5bb`
