# R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md

Self-contained reproducibility bundle. Every artifact referenced by the manuscript is included below by bundle-relative path plus SHA256; no reader depends on author-local machine paths. The v5 data manifest records `method_predictions = 0` and `method_runs = 0`; the v5 collector is data-only by construction and imports no method code. The development-time artifacts (confirmatory statistics and verifier, the v3 failure record, the Cross AML prototype) are included as claimed in the manuscript availability section.

```json
{
  "bundle": "R5C reproducibility bundle (self-contained; all paths below are bundle-relative)",
  "generated_at": "2026-09-11",
  "method_execution_evidence": {
    "v5_method_predictions": 0,
    "v5_method_runs": 0,
    "evidence_path": "v5_corpus/V5_DATA_MANIFEST.json",
    "collector_sha256": "9CF0F34F8E0935FA746D5D4F9DBBEFE70AA6B10F5F4E6F099B0D93C72ADBFEE6"
  },
  "entries": [
    {
      "bundle_path": "v5_preregistration/V5_DESIGN_OVERVIEW.md",
      "sha256": "1C14FDB6C3DB52A70623093D51B00C1CAD673B8F515F300DF4224CE0DDE0F135",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_DESIGN_OVERVIEW.md",
      "size_bytes": 8394
    },
    {
      "bundle_path": "v5_preregistration/V5_CORPUS_SPEC.md",
      "sha256": "48D87BC50CF9D2CA195DFBC0F206BC7BB040D0B00F580A37A8FCFB1F46E25795",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_CORPUS_SPEC.md",
      "size_bytes": 6503
    },
    {
      "bundle_path": "v5_preregistration/V5_DISJOINTNESS_PROTOCOL.md",
      "sha256": "A00A08AD039B5F68D3BF089F8E5C6A7FD4D7A5707BB86A39F20327CC4DC8DBA2",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_DISJOINTNESS_PROTOCOL.md",
      "size_bytes": 3725
    },
    {
      "bundle_path": "v5_preregistration/V5_DATA_ADEQUACY_DESIGN.md",
      "sha256": "A89AE546FB23830C38C97A18AC64F69368B782B237EC4A3BC7056ECD17376474",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_DATA_ADEQUACY_DESIGN.md",
      "size_bytes": 3994
    },
    {
      "bundle_path": "v5_preregistration/V5_DATA_ACCRUAL_PREREGISTRATION.md",
      "sha256": "EE25621627ED707DEBC69E828622C783F667D012A81F8681DE03C61B64A1D1F3",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_DATA_ACCRUAL_PREREGISTRATION.md",
      "size_bytes": 5001
    },
    {
      "bundle_path": "v5_preregistration/V5_COUNTING_DEFINITIONS.md",
      "sha256": "FC6544B6EA86F9D73A99A46E19930C53EF409FDC40BF078F980AA802E3875271",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_COUNTING_DEFINITIONS.md",
      "size_bytes": 3012
    },
    {
      "bundle_path": "v5_preregistration/V5_EXECUTION_MACHINERY.md",
      "sha256": "F3B2BF00C02C2EBC7AA99B64E99A7F5D815B17DBCB573EF291337DC88F9EA3C8",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_EXECUTION_MACHINERY.md",
      "size_bytes": 4712
    },
    {
      "bundle_path": "v5_preregistration/V5_HASH_MANIFEST.json",
      "sha256": "639ABD588C0CD5AEB9F252969FFC5CF5CAC5C13EBA8343B8B7C76D612503F237",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/V5_HASH_MANIFEST.json",
      "size_bytes": 2747
    },
    {
      "bundle_path": "v5_preregistration/prior_art_controls.py",
      "sha256": "C4902C6BD18513FF69DF1034FFE4EF54244FE44A933A28BE5FD16E69281DEECD",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/prior_art_controls.py",
      "size_bytes": 2938
    },
    {
      "bundle_path": "v5_preregistration/PRIOR_ART_CONTROL_HASHES.json",
      "sha256": "B383FBBFF6EBEEA82FA0780D0C4127738018E8EBB8751A792E62F04DBBB650E9",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_r5_external_validation/PRIOR_ART_CONTROL_HASHES.json",
      "size_bytes": 626
    },
    {
      "bundle_path": "v5_corpus/V5_DATA_MANIFEST.json",
      "sha256": "4389FFC8790B67B8835706B4994EE9D22530239796792BE6FA003C45E73A88B1",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/V5_DATA_MANIFEST.json",
      "size_bytes": 661
    },
    {
      "bundle_path": "v5_corpus/V5_DISJOINTNESS_REPORT.md",
      "sha256": "D51061B1B9D3947CE2A09AC8701BD4FA09DEA5364F87B6D24319CF827C7DB36A",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/V5_DISJOINTNESS_REPORT.md",
      "size_bytes": 10767
    },
    {
      "bundle_path": "v5_corpus/V5_DATA_ADEQUACY_REPORT.md",
      "sha256": "B8F0DAAE5284D31378D8DB61714E2698860086EF0FAAED601124D574E0600EF0",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/V5_DATA_ADEQUACY_REPORT.md",
      "size_bytes": 732
    },
    {
      "bundle_path": "v5_corpus/V5_CLUSTER_SUMMARY.csv",
      "sha256": "7772B71337426294485D30F475E8944BC8E22E1B95798403C2DD2962BC40ED44",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/V5_CLUSTER_SUMMARY.csv",
      "size_bytes": 468
    },
    {
      "bundle_path": "v5_corpus/V5_COLLECTION_LOG.md",
      "sha256": "543B69760222D6834965F7003D5187C5AE71388E4F0EA63FCF419D2417DCA781",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/V5_COLLECTION_LOG.md",
      "size_bytes": 7353
    },
    {
      "bundle_path": "v5_corpus/cumulative_adequacy.json",
      "sha256": "1D28ADDEBE8379C6C528D97870F74C28617382C74534F6D0057E8A857CB4E9E4",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/cumulative_adequacy.json",
      "size_bytes": 10402
    },
    {
      "bundle_path": "v5_corpus/collector/v5_preflight.py",
      "sha256": "9CF0F34F8E0935FA746D5D4F9DBBEFE70AA6B10F5F4E6F099B0D93C72ADBFEE6",
      "origin_workspace_path": "scripts/multi_bridge/tifs_external/v5_preflight.py",
      "size_bytes": 25603
    },
    {
      "bundle_path": "v4_summary/v4_final_report.json",
      "sha256": "EDE473B68984B4B7BA7BD5CE0B327AA377644FF7AFC47F6C2305694AFC5791B6",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v4/v4_final_report.json",
      "size_bytes": 1399
    },
    {
      "bundle_path": "v4_summary/cumulative_adequacy.json",
      "sha256": "EAFABD3BFFBCC24377E2BBAC8126F0200E78FCD347BFF46E91BF5AF41659AB49",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v4/cumulative_adequacy.json",
      "size_bytes": 6062
    },
    {
      "bundle_path": "r5c_audit/R5C_V5_ADEQUACY_FORMULA_AUDIT.md",
      "sha256": "95932740F195EFB081496F8506DFA252F538E66685DE61119723B05556B52D21",
      "origin_workspace_path": "3/chinese_rewrite_r5/audit/R5C_V5_ADEQUACY_FORMULA_AUDIT.md",
      "size_bytes": 4226
    },
    {
      "bundle_path": "r5c_audit/R5C_PRIOR_ART_CORRECTION_NOTE.md",
      "sha256": "079979ADB1237CAE0BE00999BFD353D960EB4BD98F1921E7B6C470D7A34B393A",
      "origin_workspace_path": "3/chinese_rewrite_r5/audit/R5C_PRIOR_ART_CORRECTION_NOTE.md",
      "size_bytes": 3189
    },
    {
      "bundle_path": "r5c_audit/R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md",
      "sha256": "746B8C81C1F8DFDAB8FDD60D2C9D67432660BFD190E14EFB320DA6B41B71A3B8",
      "origin_workspace_path": "3/chinese_rewrite_r5/audit/R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md",
      "size_bytes": 8022
    },
    {
      "bundle_path": "scalability/R5_SCALABILITY_RESULTS.csv",
      "sha256": "1C11F59844BA027C7DDDF6DD38D51E9428608DEB37BB57D0BA5B1C24FF1F8B32",
      "origin_workspace_path": "3/chinese_rewrite_r5/results/scalability/R5_SCALABILITY_RESULTS.csv",
      "size_bytes": 13033
    },
    {
      "bundle_path": "scalability/R5_SCALABILITY_REPORT.md",
      "sha256": "870DA82B986C2F45B30E380364BE2336CD5A78CE16E4C0017C727637B945C57C",
      "origin_workspace_path": "3/chinese_rewrite_r5/results/scalability/R5_SCALABILITY_REPORT.md",
      "size_bytes": 3632
    },
    {
      "bundle_path": "confirmatory_artifacts/statistics.json",
      "sha256": "43D98964046ED8C51DFEE50327E1DBDEB65AAA894A3809FA44932737E49B4517",
      "origin_workspace_path": "out/multi_bridge_expansion/conditional_plan_holdout_results/statistics.json",
      "size_bytes": 4706
    },
    {
      "bundle_path": "confirmatory_artifacts/verification.json",
      "sha256": "96931E28982C84B0C14872BD33542191BC043EA9362586AFEAFB4EB681040063",
      "origin_workspace_path": "out/multi_bridge_expansion/conditional_plan_holdout_results/verification.json",
      "size_bytes": 747
    },
    {
      "bundle_path": "confirmatory_artifacts/run_locked_holdout.py",
      "sha256": "EB48C35E78E30443AC4EC4D44434B1D9E060121FCEE5D6CEC270D71A10F6ABEE",
      "origin_workspace_path": "scripts/multi_bridge/holdout/run_locked_holdout.py",
      "size_bytes": 17003
    },
    {
      "bundle_path": "confirmatory_artifacts/verify_locked_temporal_external_validation.py",
      "sha256": "833D955AA1C59654691D498B588B96405FC2F2EC7BE90A109FFAD5D177C83240",
      "origin_workspace_path": "scripts/multi_bridge/tifs_external/verify_locked_temporal_external_validation.py",
      "size_bytes": 2047
    },
    {
      "bundle_path": "v3_record/V3_EXECUTION_FAILURE_ADJUDICATION.md",
      "sha256": "FD011E585EAA0F627A121BC554CE6074CC25B678E2A21BA4964D41A6C9E8C46B",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_validation_preregistration_v3/V3_EXECUTION_FAILURE_ADJUDICATION.md",
      "size_bytes": 1664
    },
    {
      "bundle_path": "v5_corpus/blocks/b01/anchors.json",
      "sha256": "27F76E4548FD2E3797CB8E2F230D6B66E7C13A07FE839C428E0F3F7A2C8802FE",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b01/anchors.json",
      "size_bytes": 1859591
    },
    {
      "bundle_path": "v5_corpus/blocks/b02/anchors.json",
      "sha256": "8A762DBB30AF92EB59E916BC3D438882D257777825D43461E236919CF93BD104",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b02/anchors.json",
      "size_bytes": 3139630
    },
    {
      "bundle_path": "v5_corpus/blocks/b03/anchors.json",
      "sha256": "926C42BE078F15F96EBFA3BF7DF91156E51AE6B302ED4BED560C937C383A30D8",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b03/anchors.json",
      "size_bytes": 3818473
    },
    {
      "bundle_path": "v5_corpus/blocks/b04/anchors.json",
      "sha256": "E14F98D53D091A323B6D4EE162D7E02F599756F0B7FCEAAB32BBC6A03A3B1231",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b04/anchors.json",
      "size_bytes": 3331667
    },
    {
      "bundle_path": "v5_corpus/blocks/b05/anchors.json",
      "sha256": "F6CDA2ED148D54479EDE7B7FD475FA0F29BED6674724C7DBE495AD1D3C479E25",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b05/anchors.json",
      "size_bytes": 9393070
    },
    {
      "bundle_path": "v5_corpus/blocks/b06/anchors.json",
      "sha256": "CE7B02D649B4B034E53989417C642A15C3277DE671809A4651CDA94A22D9E73A",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b06/anchors.json",
      "size_bytes": 5440535
    },
    {
      "bundle_path": "v5_corpus/blocks/b07/anchors.json",
      "sha256": "92F7331B54588AA130A6B7187F99596D478FEBA6C5BB20BAEBCBB457CB28214F",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b07/anchors.json",
      "size_bytes": 3026718
    },
    {
      "bundle_path": "v5_corpus/blocks/b08/anchors.json",
      "sha256": "EAC79FD2AF6FEFBE1D4CB00A5520D05426C6BAFFEE3A9692B9453DFEC7936F1A",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b08/anchors.json",
      "size_bytes": 2205824
    },
    {
      "bundle_path": "v5_corpus/blocks/b09/anchors.json",
      "sha256": "1E2C7B1BEA9CC8946E6AB6ADE8543186B8D05D987006D12FD2D7365F0D4A66A9",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b09/anchors.json",
      "size_bytes": 2584702
    },
    {
      "bundle_path": "v5_corpus/blocks/b10/anchors.json",
      "sha256": "0415548C778B2AF5E448E589CDC41E7B03ADF782509DB4FA3749C7CB2175A7E6",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b10/anchors.json",
      "size_bytes": 2477936
    },
    {
      "bundle_path": "v5_corpus/blocks/b11/anchors.json",
      "sha256": "6334337F20601B78C8775354F7568BD29367914352C6B122FF4C8F1C206D57E3",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b11/anchors.json",
      "size_bytes": 2378938
    },
    {
      "bundle_path": "v5_corpus/blocks/b12/anchors.json",
      "sha256": "C542DE803CC506B7DA6B519F07F847383611FB681BCDB72E8AC96A9F220F2FDC",
      "origin_workspace_path": "out/multi_bridge_expansion/tifs_temporal_external_v5/blocks/b12/anchors.json",
      "size_bytes": 1887076
    },
    {
      "bundle_path": "cross_aml/__init__.py",
      "sha256": "EAEF4D07EDD3F7B55352C35AB4D544DCD19B71CC55399C196A0780F750A4A89C",
      "origin_workspace_path": "tools/cross_aml/cross_aml/__init__.py",
      "size_bytes": 107
    },
    {
      "bundle_path": "cross_aml/abstention.py",
      "sha256": "112EF5DC27ACE7409A2FC1F52F945BE8224E915C83AE3113D3546604BFD21FBC",
      "origin_workspace_path": "tools/cross_aml/cross_aml/abstention.py",
      "size_bytes": 1691
    },
    {
      "bundle_path": "cross_aml/bridge_parser.py",
      "sha256": "34614AB2134300B2FBCCBB28AD38A2E55110290AE502A29C898F64B10DBA5333",
      "origin_workspace_path": "tools/cross_aml/cross_aml/bridge_parser.py",
      "size_bytes": 4799
    },
    {
      "bundle_path": "cross_aml/chains.py",
      "sha256": "D859F7588AE9A33D428CA5576ED30D78E94C9303007C13EAF8A1C9747F56A6E8",
      "origin_workspace_path": "tools/cross_aml/cross_aml/chains.py",
      "size_bytes": 925
    },
    {
      "bundle_path": "cross_aml/cli.py",
      "sha256": "DD274EF70955E01F450DE93B394E1979C5F824D5DA9ABE34E580DE4DF5170492",
      "origin_workspace_path": "tools/cross_aml/cross_aml/cli.py",
      "size_bytes": 8191
    },
    {
      "bundle_path": "cross_aml/config.py",
      "sha256": "A82700AFA6AC43E9E8203A7B54C2C7D7A5A5E67E0D05EB374B2DD9C367D6400C",
      "origin_workspace_path": "tools/cross_aml/cross_aml/config.py",
      "size_bytes": 2042
    },
    {
      "bundle_path": "cross_aml/coverage.py",
      "sha256": "13037DB7C7876F302A6BEBC92EC19B972C5198151030FBCF849C29E12D694B57",
      "origin_workspace_path": "tools/cross_aml/cross_aml/coverage.py",
      "size_bytes": 2531
    },
    {
      "bundle_path": "cross_aml/discovery_limits.py",
      "sha256": "DAA35676C334409BA08800B0B20AE58B407023C6EB7B5C74873F812613B90E37",
      "origin_workspace_path": "tools/cross_aml/cross_aml/discovery_limits.py",
      "size_bytes": 2032
    },
    {
      "bundle_path": "cross_aml/event_fetcher.py",
      "sha256": "E312CFF53B34BE35BB637B17492524EC71D935B7CB0A8DDEA821C46CEC672FF4",
      "origin_workspace_path": "tools/cross_aml/cross_aml/event_fetcher.py",
      "size_bytes": 2810
    },
    {
      "bundle_path": "cross_aml/explanation.py",
      "sha256": "CDD23E78BB3A88B071BB7E519ED8C8507C776C22DDD7CA76E88AB2A030A00168",
      "origin_workspace_path": "tools/cross_aml/cross_aml/explanation.py",
      "size_bytes": 2474
    },
    {
      "bundle_path": "cross_aml/feature_builder.py",
      "sha256": "C0ACC1D08C3EA461F952E79D3FCD00A347831CFEC8560593DE5D436EE1B0B389",
      "origin_workspace_path": "tools/cross_aml/cross_aml/feature_builder.py",
      "size_bytes": 3470
    },
    {
      "bundle_path": "cross_aml/flow_builder.py",
      "sha256": "FB93BE9EEEF7C49C66D96A990CDA35D7456BCAD759EBE22C45A8FFC22DF6FAC2",
      "origin_workspace_path": "tools/cross_aml/cross_aml/flow_builder.py",
      "size_bytes": 4861
    },
    {
      "bundle_path": "cross_aml/mock_data.py",
      "sha256": "5F9FFB6D6DD50E5C11D1AD9673461A28FCFAB779D05DDEA0F3D795513BDA93C6",
      "origin_workspace_path": "tools/cross_aml/cross_aml/mock_data.py",
      "size_bytes": 3790
    },
    {
      "bundle_path": "cross_aml/quotient_builder.py",
      "sha256": "9F112E7262CCC3158F344FFBE6E56CD5A1CB142C76842C9DB20E2868DD6CEAA2",
      "origin_workspace_path": "tools/cross_aml/cross_aml/quotient_builder.py",
      "size_bytes": 1875
    },
    {
      "bundle_path": "cross_aml/rcuotq_matcher.py",
      "sha256": "C48B6C7928954D801A06F6EA314DDFB9E8A31A00DA207A62D18BAB1BA9DC7A2F",
      "origin_workspace_path": "tools/cross_aml/cross_aml/rcuotq_matcher.py",
      "size_bytes": 3199
    },
    {
      "bundle_path": "cross_aml/report_writer.py",
      "sha256": "D51686A42104AC88EC727B2C8E8254D9DEBB3E2A0102CC5D637371F6E30FDBCF",
      "origin_workspace_path": "tools/cross_aml/cross_aml/report_writer.py",
      "size_bytes": 6178
    },
    {
      "bundle_path": "cross_aml/rpc_client.py",
      "sha256": "F19D244D0BA326FF720F9073BF1DF55F21676EB1CA61CD72F93DDB4389A3C562",
      "origin_workspace_path": "tools/cross_aml/cross_aml/rpc_client.py",
      "size_bytes": 3639
    },
    {
      "bundle_path": "cross_aml/schemas.py",
      "sha256": "4DD0DDE17BFFD061DF888A0972D4C921834874038C77C2E5A782F04FAA8ECB0D",
      "origin_workspace_path": "tools/cross_aml/cross_aml/schemas.py",
      "size_bytes": 2634
    },
    {
      "bundle_path": "cross_aml/token_transfer_parser.py",
      "sha256": "B0877285EB2032C96309B7E8177C4EC5CD6C9FDC786DB6E0C67370C7431FCDA1",
      "origin_workspace_path": "tools/cross_aml/cross_aml/token_transfer_parser.py",
      "size_bytes": 3260
    },
    {
      "bundle_path": "cross_aml/utils.py",
      "sha256": "312BB868CA0E0523889DE4AFB8671AD75BE09A6A7296C20BF449710D99C79D70",
      "origin_workspace_path": "tools/cross_aml/cross_aml/utils.py",
      "size_bytes": 1720
    },
    {
      "bundle_path": "cross_aml/run_cross_aml.py",
      "sha256": "F2E0B045E668CF37112F0F4B3E3BD6CBEA888608B9EA566EED4AB760754F3C23",
      "origin_workspace_path": "tools/cross_aml/run_cross_aml.py",
      "size_bytes": 354
    },
    {
      "bundle_path": "cross_aml/test_abstention.py",
      "sha256": "012D3CA23990911F44C10E04424BAEFF3158153321D33981BB7716A323E08D3D",
      "origin_workspace_path": "tools/cross_aml/tests/test_abstention.py",
      "size_bytes": 1222
    },
    {
      "bundle_path": "cross_aml/test_config.py",
      "sha256": "0276AA2C047A9D0824F5D05710FB67BDE649B1604FF5691E70227D162738E6F8",
      "origin_workspace_path": "tools/cross_aml/tests/test_config.py",
      "size_bytes": 1031
    },
    {
      "bundle_path": "cross_aml/test_coverage.py",
      "sha256": "E4475A0B920BB4BFE08BAE45D9A041EC9EAA212C32AA235CB274395C869487A8",
      "origin_workspace_path": "tools/cross_aml/tests/test_coverage.py",
      "size_bytes": 1407
    },
    {
      "bundle_path": "cross_aml/test_discovery_limits.py",
      "sha256": "8E61ABEE844F726B0ED413F10DAC1B7E73D694977219BFB0AC5901894DA56872",
      "origin_workspace_path": "tools/cross_aml/tests/test_discovery_limits.py",
      "size_bytes": 864
    },
    {
      "bundle_path": "cross_aml/test_explanation.py",
      "sha256": "DDD5048426D1BDB9F0371D33368ADAEC8A6B466908718DBA0C7AF7357CE18CF8",
      "origin_workspace_path": "tools/cross_aml/tests/test_explanation.py",
      "size_bytes": 1718
    },
    {
      "bundle_path": "cross_aml/test_flow_builder.py",
      "sha256": "448D9238A0335FEA5B02F7D9E678A4F6EFC0A9B6A375CA0621130CAAFB9C3CC9",
      "origin_workspace_path": "tools/cross_aml/tests/test_flow_builder.py",
      "size_bytes": 613
    }
  ]
}
```
