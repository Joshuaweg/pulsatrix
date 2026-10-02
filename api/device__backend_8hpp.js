var device__backend_8hpp =
[
    [ "pulsatrix::RecurrentCellArgs", "structpulsatrix_1_1RecurrentCellArgs.html", "structpulsatrix_1_1RecurrentCellArgs" ],
    [ "pulsatrix::SsmPassArgs", "structpulsatrix_1_1SsmPassArgs.html", "structpulsatrix_1_1SsmPassArgs" ],
    [ "pulsatrix::RlRowArgs", "structpulsatrix_1_1RlRowArgs.html", "structpulsatrix_1_1RlRowArgs" ],
    [ "pulsatrix::DeviceBackend", "classpulsatrix_1_1DeviceBackend.html", "classpulsatrix_1_1DeviceBackend" ],
    [ "CopyDirection", "device__backend_8hpp.html#a5d9adebabff0df0875f22a763d77ad6b", [
      [ "HostToDevice", "device__backend_8hpp.html#a5d9adebabff0df0875f22a763d77ad6baa9988afceee3dbd1517b549bbe0f5e92", null ],
      [ "DeviceToHost", "device__backend_8hpp.html#a5d9adebabff0df0875f22a763d77ad6ba6170220e5a9b44706ce35a6ff1dc37fd", null ],
      [ "DeviceToDevice", "device__backend_8hpp.html#a5d9adebabff0df0875f22a763d77ad6bad9ccc4ace3b87f3b327a14e17ed5fc6a", null ],
      [ "HostToHost", "device__backend_8hpp.html#a5d9adebabff0df0875f22a763d77ad6ba297d3681581537b0677cb9f2a3b9e589", null ]
    ] ],
    [ "DeviceType", "device__backend_8hpp.html#a5480c8cbe462fe3f5a8eb04a8c579e6e", [
      [ "Cpu", "device__backend_8hpp.html#a5480c8cbe462fe3f5a8eb04a8c579e6ea54c82ef76ecbbd4c2293e09bae01b54e", null ],
      [ "Cuda", "device__backend_8hpp.html#a5480c8cbe462fe3f5a8eb04a8c579e6ea8b95dcff7397d0693c03e394af5552aa", null ],
      [ "Hip", "device__backend_8hpp.html#a5480c8cbe462fe3f5a8eb04a8c579e6eaafcd5ccb84b5c522c66efa7836e17f92", null ]
    ] ],
    [ "ElementwiseOp", "device__backend_8hpp.html#afec27fa326532e4ad62b32d54558b503", [
      [ "Relu", "device__backend_8hpp.html#afec27fa326532e4ad62b32d54558b503a7bfde445daa113a9903d4eaa43b41e2b", null ],
      [ "Neg", "device__backend_8hpp.html#afec27fa326532e4ad62b32d54558b503afb278fa5defd7e699fcbc930c3e76ccd", null ],
      [ "Tanh", "device__backend_8hpp.html#afec27fa326532e4ad62b32d54558b503acc132a41cab5676334f353a22a0aa5c5", null ],
      [ "Sigmoid", "device__backend_8hpp.html#afec27fa326532e4ad62b32d54558b503a21eebb164e4b8b9bcf64fdb4d8d5dff4", null ],
      [ "Silu", "device__backend_8hpp.html#afec27fa326532e4ad62b32d54558b503a17aeea3715b4cdfdf861f237f4011edf", null ],
      [ "Exp", "device__backend_8hpp.html#afec27fa326532e4ad62b32d54558b503acad39a154bffb61175f674d6eefaf6d0", null ]
    ] ],
    [ "LogicOp", "device__backend_8hpp.html#a23ce7985a73e7df95c9467eac40d5d9e", [
      [ "ConjunctionForward", "device__backend_8hpp.html#a23ce7985a73e7df95c9467eac40d5d9eab09c932ce0af7600b64c202bc1f7c4f3", null ],
      [ "ConjunctionBackward", "device__backend_8hpp.html#a23ce7985a73e7df95c9467eac40d5d9ea228e1fd8b35a176d23bf8f941ff0c77e", null ],
      [ "ConjunctionLrp", "device__backend_8hpp.html#a23ce7985a73e7df95c9467eac40d5d9ea4b739ab135e0fbd479f7b2c287b6e164", null ],
      [ "DisjunctionForward", "device__backend_8hpp.html#a23ce7985a73e7df95c9467eac40d5d9eac5e16d9f8dea3ad7bcadf3e06400c63c", null ],
      [ "DisjunctionBackward", "device__backend_8hpp.html#a23ce7985a73e7df95c9467eac40d5d9ea726f8215d974d8309f5e669c1c715454", null ],
      [ "DisjunctionLrp", "device__backend_8hpp.html#a23ce7985a73e7df95c9467eac40d5d9eae93a7af9d2b5ab7e9b0c969635cf7d3f", null ]
    ] ],
    [ "LrpGate", "device__backend_8hpp.html#aeafb34c87ad858963e6f06a87ee9bbc3", [
      [ "None", "device__backend_8hpp.html#aeafb34c87ad858963e6f06a87ee9bbc3a6adf97f83acf6453d4a6a4b1070f3754", null ],
      [ "Positive", "device__backend_8hpp.html#aeafb34c87ad858963e6f06a87ee9bbc3a3289297424e01eda5b788c083bbf3147", null ],
      [ "Negative", "device__backend_8hpp.html#aeafb34c87ad858963e6f06a87ee9bbc3affb9356ff2b7da85c75c92fa7ea03b8b", null ]
    ] ],
    [ "RecurrentCellOp", "device__backend_8hpp.html#ac55b0127cfe9bf4ce7793fef9db97400", [
      [ "RnnBackward", "device__backend_8hpp.html#ac55b0127cfe9bf4ce7793fef9db97400a789b7cc15a19f6069e16e2522deebf64", null ],
      [ "LstmForward", "device__backend_8hpp.html#ac55b0127cfe9bf4ce7793fef9db97400ab17570322128d1651d18c17847ead959", null ],
      [ "LstmBackward", "device__backend_8hpp.html#ac55b0127cfe9bf4ce7793fef9db97400aafa50372cfb2e47402477809626c0887", null ],
      [ "LstmLrp", "device__backend_8hpp.html#ac55b0127cfe9bf4ce7793fef9db97400a651b9b5b61c6f9c608a91c977f1d1731", null ],
      [ "GruBackward", "device__backend_8hpp.html#ac55b0127cfe9bf4ce7793fef9db97400ab250d5c399d7c2349e16a8f2ecd36185", null ],
      [ "GruLrp", "device__backend_8hpp.html#ac55b0127cfe9bf4ce7793fef9db97400aeb5497c21260ca12facce3759273ab44", null ]
    ] ],
    [ "RlRowOp", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930", [
      [ "DqnLoss", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930a968bf24ac72a6aace6ce47990ee9787f", null ],
      [ "DqnGrad", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930ab3bad3db141dca05dac2e9ec9067afe1", null ],
      [ "PgLoss", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930ab760ac927f4ec969409760e2f044e09f", null ],
      [ "PgGrad", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930adc1f63481ef485e13b060c446d273bbf", null ],
      [ "PpoLoss", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930a8bc4fa0f592f03c568063df4971d27f9", null ],
      [ "PpoGrad", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930a2c877e3b6b6c4faf78580a5ec9780536", null ],
      [ "DqnTarget", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930a97740b358c4a01f16fb9d431dc884b87", null ],
      [ "PolyakBlend", "device__backend_8hpp.html#af72bbbb89c154b30ca26561ab2f87930a1f0ee8594b99d914e20915ca8f0fa1e9", null ]
    ] ],
    [ "SsmPassOp", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883", [
      [ "MambaForward", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883a4b329f1a455bec72ccc36effa291c8a4", null ],
      [ "MambaBackward", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883aafd6ee7fcd70442a7b08a77881574ab2", null ],
      [ "MambaGradBC", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883aa4c2ce7fcd5cbf8e07420e447bac3987", null ],
      [ "MambaLrp", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883aaca452419514797385505efded2e0473", null ],
      [ "RwkvTokenShift", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883abcdd6c9e1928960ed3f5a5f38cc9b8d1", null ],
      [ "RwkvForward", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883a02074e773654d60be53588a7c551717e", null ],
      [ "RwkvBackward", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883ad67af2133293778b01e15a50778282d7", null ],
      [ "RwkvShiftBackward", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883ac4e80c16787796dda381d1515c6af876", null ],
      [ "RwkvLrp", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883ac5aa76383a18f79f619e4523207ad4d5", null ],
      [ "RwkvShiftLrp", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883a5df15a49eb81d6e7b5c201c81d93b364", null ],
      [ "StabilizedDiv", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883af7357d9f8c4df9c447eaaec3710f1a2f", null ],
      [ "RetnetForward", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883ae540a55d2cfff0c1b3c1fdaa6ea86072", null ],
      [ "RetnetStateGrad", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883a9ae91a09f05f50ad5e936dbcc8dd137a", null ],
      [ "RetnetGradQK", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883accf9209bedc3fd2059a7daa6455e147a", null ],
      [ "RetnetGradV", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883a03d4e7d6d672e547c44f37cd463c518b", null ],
      [ "RetnetScores", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883a00b785050045bc646eeac7d9b576b822", null ],
      [ "RetnetReadout", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883af885ed8e459ad9321616f3ce9e71e97f", null ],
      [ "RetnetLrpInput", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883a08ac54865c115b01837b65f79eb12f97", null ],
      [ "ReverseTimeSum", "device__backend_8hpp.html#a9ea30dd018ca9a3db2acc46af2880883abe5b38bee7a84ed6af7101830bf9cae3", null ]
    ] ]
];