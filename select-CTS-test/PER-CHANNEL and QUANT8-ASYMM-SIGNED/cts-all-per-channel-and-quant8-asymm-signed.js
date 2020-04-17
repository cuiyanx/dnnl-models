describe('CTS Supplement Test', function() {
  this.timeout(20000);
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();


  it('check result for Conv2d per channel example-1', async function() {
    // For 'Conv2d per channel' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [138, 138, 138, 138, 138, 138];
    let op4_expect = [137, 141, 145, 137, 141, 145, 137, 141, 145];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op3 = operandIndex++;
    model.addOperand(type2);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op3, new Int32Array([4, 4, 4]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d per channel example-2', async function() {
    // For 'Conv2d per channel' example: examples_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [138, 138, 138, 138, 138, 138];
    let op2_value = [1, 2, 1, 2, 1, 2];
    let op3_value = [4, 4, 4];
    let op4_expect = [137, 141, 145, 137, 141, 145, 137, 141, 145];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type18 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type18_length = product(type18.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type18);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op3 = operandIndex++;
    model.addOperand(type2);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d per channel example-3', async function() {
    // For 'Conv2d per channel' example: examples_layouts_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [138, 108, 138, 108, 138, 108];
    let op41_expect = [121, 118, 115, 121, 118, 115, 121, 118, 115];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type6 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type0);
    let op21 = operandIndex++;
    model.addOperand(type6);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op31 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op21, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op31, new Int32Array([4, 4, 4]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([1]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10, param11, param12, param13], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type3_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d per channel example-4', async function() {
    // For 'Conv2d per channel' example: examples_layouts_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [138, 108, 138, 108, 138, 108];
    let op21_value = [1, 2, 1, 2, 1, 2];
    let op31_value = [4, 4, 4];
    let op41_expect = [121, 118, 115, 121, 118, 115, 121, 118, 115];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type19 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type19_length = product(type19.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op11 = operandIndex++;
    model.addOperand(type0);
    let op21 = operandIndex++;
    model.addOperand(type19);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op31 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([0]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([1]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10, param11, param12, param13], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type3_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });


  it('check result for Conv2d quant8 signed example-1', async function() {
    // For 'Conv2d quant8 signed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [10, 10, 10, 10, 10, 10];
    let op45_expect = [9, 13, 17, 9, 13, 17, 9, 13, 17];

    let type10 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 0};
    let type10_length = product(type10.dimensions);
    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 0};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type9_length = product(type9.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type7);
    let op25 = operandIndex++;
    model.addOperand(type8);
    model.setOperandSymmPerChannelQuantParams(op25, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op35 = operandIndex++;
    model.addOperand(type9);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let param41 = operandIndex++;
    model.addOperand(type4);
    let param42 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op25, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op35, new Int32Array([4, 4, 4]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.setOperandValue(param37, new Int32Array([0]));
    model.setOperandValue(param38, new Int32Array([0]));
    model.setOperandValue(param39, new Int32Array([0]));
    model.setOperandValue(param40, new Int32Array([1]));
    model.setOperandValue(param41, new Int32Array([1]));
    model.setOperandValue(param42, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param36, param37, param38, param39, param40, param41, param42], [op45]);

    model.identifyInputsAndOutputs([op15], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Int8Array(op15_value);
    execution.setInput(0, op15_input);
    let op45_output = new Int8Array(type10_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-2', async function() {
    // For 'Conv2d quant8 signed' example: examples_layouts_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op16_value = [10, -20, 10, -20, 10, -20];
    let op46_expect = [-7, -10, -13, -7, -10, -13, -7, -10, -13];

    let type10 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 0};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type11_length = product(type11.dimensions);
    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 0};
    let type7_length = product(type7.dimensions);
    let type9 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type9_length = product(type9.dimensions);

    let op16 = operandIndex++;
    model.addOperand(type7);
    let op26 = operandIndex++;
    model.addOperand(type11);
    model.setOperandSymmPerChannelQuantParams(op26, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op36 = operandIndex++;
    model.addOperand(type9);
    let param43 = operandIndex++;
    model.addOperand(type4);
    let param44 = operandIndex++;
    model.addOperand(type4);
    let param45 = operandIndex++;
    model.addOperand(type4);
    let param46 = operandIndex++;
    model.addOperand(type4);
    let param47 = operandIndex++;
    model.addOperand(type4);
    let param48 = operandIndex++;
    model.addOperand(type4);
    let param49 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op26, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op36, new Int32Array([4, 4, 4]));
    model.setOperandValue(param43, new Int32Array([0]));
    model.setOperandValue(param44, new Int32Array([0]));
    model.setOperandValue(param45, new Int32Array([0]));
    model.setOperandValue(param46, new Int32Array([0]));
    model.setOperandValue(param47, new Int32Array([1]));
    model.setOperandValue(param48, new Int32Array([1]));
    model.setOperandValue(param49, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op16, op26, op36, param43, param44, param45, param46, param47, param48, param49], [op46]);

    model.identifyInputsAndOutputs([op16], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op16_input = new Int8Array(op16_value);
    execution.setInput(0, op16_input);
    let op46_output = new Int8Array(type10_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-3', async function() {
    // For 'Conv2d quant8 signed' example: examples_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op17_value = [-126, -126, -126, -126, -127, -126, -126, -126, -126];
    let op47_expect = [-121, -121, -121, -121];

    let type4 = {type: nn.INT32};
    let type51 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: -128};
    let type51_length = product(type51.dimensions);
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: -128};
    let type54_length = product(type54.dimensions);
    let type72 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 1]};
    let type72_length = product(type72.dimensions);
    let type73 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type73_length = product(type73.dimensions);

    let op17 = operandIndex++;
    model.addOperand(type51);
    let op27 = operandIndex++;
    model.addOperand(type72);
    model.setOperandSymmPerChannelQuantParams(op27, {channelDim: 0, scales: new Float32Array([0.125])});
    let op37 = operandIndex++;
    model.addOperand(type73);
    let param70 = operandIndex++;
    model.addOperand(type4);
    let param71 = operandIndex++;
    model.addOperand(type4);
    let param72 = operandIndex++;
    model.addOperand(type4);
    let param73 = operandIndex++;
    model.addOperand(type4);
    let param74 = operandIndex++;
    model.addOperand(type4);
    let param75 = operandIndex++;
    model.addOperand(type4);
    let param76 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op27, new Int8Array([2, 2, 2, 2]));
    model.setOperandValue(op37, new Int32Array([0]));
    model.setOperandValue(param70, new Int32Array([0]));
    model.setOperandValue(param71, new Int32Array([0]));
    model.setOperandValue(param72, new Int32Array([0]));
    model.setOperandValue(param73, new Int32Array([0]));
    model.setOperandValue(param74, new Int32Array([1]));
    model.setOperandValue(param75, new Int32Array([1]));
    model.setOperandValue(param76, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op17, op27, op37, param70, param71, param72, param73, param74, param75, param76], [op47]);

    model.identifyInputsAndOutputs([op17], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op17_input = new Int8Array(op17_value);
    execution.setInput(0, op17_input);
    let op47_output = new Int8Array(type54_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-4', async function() {
    // For 'Conv2d quant8 signed' example: examples_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op18_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
    let op48_expect = [-78, -78, -78, -78, -43, 34, 79, -78, -78, -44, -17, -78];

    let type4 = {type: nn.INT32};
    let type74 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: -1};
    let type74_length = product(type74.dimensions);
    let type76 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: -78};
    let type76_length = product(type76.dimensions);
    let type77 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type77_length = product(type77.dimensions);
    let type78 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type78_length = product(type78.dimensions);

    let op18 = operandIndex++;
    model.addOperand(type74);
    let op28 = operandIndex++;
    model.addOperand(type77);
    model.setOperandSymmPerChannelQuantParams(op28, {channelDim: 0, scales: new Float32Array([0.5])});
    let op38 = operandIndex++;
    model.addOperand(type78);
    let param77 = operandIndex++;
    model.addOperand(type4);
    let param78 = operandIndex++;
    model.addOperand(type4);
    let param79 = operandIndex++;
    model.addOperand(type4);
    let param80 = operandIndex++;
    model.addOperand(type4);
    let op48 = operandIndex++;
    model.addOperand(type76);

    model.setOperandValue(op28, new Int8Array([2, 8, 14, 4, 10, 16, 6, 12, 18]));
    model.setOperandValue(op38, new Int32Array([-800]));
    model.setOperandValue(param77, new Int32Array([1]));
    model.setOperandValue(param78, new Int32Array([1]));
    model.setOperandValue(param79, new Int32Array([1]));
    model.setOperandValue(param80, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op18, op28, op38, param77, param78, param79, param80], [op48]);

    model.identifyInputsAndOutputs([op18], [op48]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op18_input = new Int8Array(op18_value);
    execution.setInput(0, op18_input);
    let op48_output = new Int8Array(type76_length);
    execution.setOutput(0, op48_output);

    await execution.startCompute();

    for (let i = 0; i < type76_length; ++i) {
      assert.isTrue(almostEqualCTS(op48_output[i], op48_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-5', async function() {
    // For 'Conv2d quant8 signed' example: examples_channel_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op19_value = [-118, -118, -118];
    let op49_expect = [-98, -53, -8];

    let type4 = {type: nn.INT32};
    let type45 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: -128};
    let type45_length = product(type45.dimensions);
    let type82 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type82_length = product(type82.dimensions);
    let type83 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type83_length = product(type83.dimensions);

    let op19 = operandIndex++;
    model.addOperand(type45);
    let op29 = operandIndex++;
    model.addOperand(type82);
    model.setOperandSymmPerChannelQuantParams(op29, {channelDim: 0, scales: new Float32Array([0.5, 0.4, 0.3])});
    let op39 = operandIndex++;
    model.addOperand(type83);
    let param81 = operandIndex++;
    model.addOperand(type4);
    let param82 = operandIndex++;
    model.addOperand(type4);
    let param83 = operandIndex++;
    model.addOperand(type4);
    let param84 = operandIndex++;
    model.addOperand(type4);
    let param85 = operandIndex++;
    model.addOperand(type4);
    let param86 = operandIndex++;
    model.addOperand(type4);
    let param87 = operandIndex++;
    model.addOperand(type4);
    let op49 = operandIndex++;
    model.addOperand(type45);

    model.setOperandValue(op29, new Int8Array([1, 2, 3, 5, 6, 8, 12, 13, 15]));
    model.setOperandValue(op39, new Int32Array([0, 0, 0]));
    model.setOperandValue(param81, new Int32Array([0]));
    model.setOperandValue(param82, new Int32Array([0]));
    model.setOperandValue(param83, new Int32Array([0]));
    model.setOperandValue(param84, new Int32Array([0]));
    model.setOperandValue(param85, new Int32Array([1]));
    model.setOperandValue(param86, new Int32Array([1]));
    model.setOperandValue(param87, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op19, op29, op39, param81, param82, param83, param84, param85, param86, param87], [op49]);

    model.identifyInputsAndOutputs([op19], [op49]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op19_input = new Int8Array(op19_value);
    execution.setInput(0, op19_input);
    let op49_output = new Int8Array(type45_length);
    execution.setOutput(0, op49_output);

    await execution.startCompute();

    for (let i = 0; i < type45_length; ++i) {
      assert.isTrue(almostEqualCTS(op49_output[i], op49_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-6', async function() {
    // For 'Conv2d quant8 signed' example: examples_large_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op110_value = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36];
    let op410_expect = [-113, -110, -107, -95, -88, -80, -77, -65, -53, -59, -42, -26, -41, -20, 1, -23, 2, 28];

    let type4 = {type: nn.INT32};
    let type86 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 0};
    let type86_length = product(type86.dimensions);
    let type88 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: -128};
    let type88_length = product(type88.dimensions);
    let type89 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type89_length = product(type89.dimensions);
    let type90 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type90_length = product(type90.dimensions);

    let op110 = operandIndex++;
    model.addOperand(type86);
    let op210 = operandIndex++;
    model.addOperand(type89);
    model.setOperandSymmPerChannelQuantParams(op210, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 0.5])});
    let op310 = operandIndex++;
    model.addOperand(type90);
    let param88 = operandIndex++;
    model.addOperand(type4);
    let param89 = operandIndex++;
    model.addOperand(type4);
    let param90 = operandIndex++;
    model.addOperand(type4);
    let param91 = operandIndex++;
    model.addOperand(type4);
    let param92 = operandIndex++;
    model.addOperand(type4);
    let param93 = operandIndex++;
    model.addOperand(type4);
    let param94 = operandIndex++;
    model.addOperand(type4);
    let op410 = operandIndex++;
    model.addOperand(type88);

    model.setOperandValue(op210, new Int8Array([2, 8, 14, 2, 5, 8, 6, 12, 18]));
    model.setOperandValue(op310, new Int32Array([0, 0, 0]));
    model.setOperandValue(param88, new Int32Array([0]));
    model.setOperandValue(param89, new Int32Array([0]));
    model.setOperandValue(param90, new Int32Array([0]));
    model.setOperandValue(param91, new Int32Array([0]));
    model.setOperandValue(param92, new Int32Array([1]));
    model.setOperandValue(param93, new Int32Array([1]));
    model.setOperandValue(param94, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op110, op210, op310, param88, param89, param90, param91, param92, param93, param94], [op410]);

    model.identifyInputsAndOutputs([op110], [op410]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op110_input = new Int8Array(op110_value);
    execution.setInput(0, op110_input);
    let op410_output = new Int8Array(type88_length);
    execution.setOutput(0, op410_output);

    await execution.startCompute();

    for (let i = 0; i < type88_length; ++i) {
      assert.isTrue(almostEqualCTS(op410_output[i], op410_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-7', async function() {
    // For 'Conv2d quant8 signed' example: examples_large_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op110_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let op410_expect = [29, 35, 41, 65, 80, 95, 101, 125, 149, 137, 170, 203, 173, 215, 257, 209, 260, 311];

    let type4 = {type: nn.INT32};
    let type91 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 3, 3], scale: 1.0, zeroPoint: -1};
    let type91_length = product(type91.dimensions);
    let type92 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type92_length = product(type92.dimensions);
    let type93 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type93_length = product(type93.dimensions);

    let op110 = operandIndex++;
    model.addOperand(type91);
    let op210 = operandIndex++;
    model.addOperand(type92);
    model.setOperandSymmPerChannelQuantParams(op210, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 1.005])});
    let op310 = operandIndex++;
    model.addOperand(type93);
    let param88 = operandIndex++;
    model.addOperand(type4);
    let param89 = operandIndex++;
    model.addOperand(type4);
    let param90 = operandIndex++;
    model.addOperand(type4);
    let param91 = operandIndex++;
    model.addOperand(type4);
    let param92 = operandIndex++;
    model.addOperand(type4);
    let param93 = operandIndex++;
    model.addOperand(type4);
    let param94 = operandIndex++;
    model.addOperand(type4);
    let op410 = operandIndex++;
    model.addOperand(type91);

    model.setOperandValue(op210, new Int8Array([2, 8, 14, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op310, new Int32Array([0, 0, 0]));
    model.setOperandValue(param88, new Int32Array([0]));
    model.setOperandValue(param89, new Int32Array([0]));
    model.setOperandValue(param90, new Int32Array([0]));
    model.setOperandValue(param91, new Int32Array([0]));
    model.setOperandValue(param92, new Int32Array([1]));
    model.setOperandValue(param93, new Int32Array([1]));
    model.setOperandValue(param94, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op110, op210, op310, param88, param89, param90, param91, param92, param93, param94], [op410]);

    model.identifyInputsAndOutputs([op110], [op410]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op110_input = new Int8Array(op110_value);
    execution.setInput(0, op110_input);
    let op410_output = new Int8Array(type91_length);
    execution.setOutput(0, op410_output);

    await execution.startCompute();

    for (let i = 0; i < type91_length; ++i) {
      assert.isTrue(almostEqualCTS(op410_output[i], op410_expect[i]));
    }
  });


  it('check result for Conv2d v1_2 example-1', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type36_length = product(type36.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.125])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array([2, 2, 2, 2]));
    model.setOperandValue(op3, new Int32Array([0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-2', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op2_value = [2, 2, 2, 2];
    let op3_value = [0];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type36_length = product(type36.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.125])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-3', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);
    let type50 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type50_length = product(type50.dimensions);
    let type51 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type51_length = product(type51.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type50);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5])});
    let op31 = operandIndex++;
    model.addOperand(type51);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Int8Array([2, 8, 14, 4, 10, 16, 6, 12, 18]));
    model.setOperandValue(op31, new Int32Array([-800]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-4', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op21_value = [2, 8, 14, 4, 10, 16, 6, 12, 18];
    let op31_value = [-800];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);
    let type50 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type50_length = product(type50.dimensions);
    let type51 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type51_length = product(type51.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type50);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5])});
    let op31 = operandIndex++;
    model.addOperand(type51);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-5', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type60 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type60_length = product(type60.dimensions);
    let type61 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type61_length = product(type61.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type60);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 0, scales: new Float32Array([0.5, 0.4, 0.3])});
    let op32 = operandIndex++;
    model.addOperand(type61);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Int8Array([1, 2, 3, 5, 6, 8, 12, 13, 15]));
    model.setOperandValue(op32, new Int32Array([0, 0, 0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-6', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op22_value = [1, 2, 3, 5, 6, 8, 12, 13, 15];
    let op32_value = [0, 0, 0];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type60 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type60_length = product(type60.dimensions);
    let type61 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type61_length = product(type61.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type60);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 0, scales: new Float32Array([0.5, 0.4, 0.3])});
    let op32 = operandIndex++;
    model.addOperand(type61);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Int8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-7', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);
    let type71 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type71_length = product(type71.dimensions);
    let type72 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type72_length = product(type72.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type71);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 0.5])});
    let op33 = operandIndex++;
    model.addOperand(type72);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Int8Array([2, 8, 14, 2, 5, 8, 6, 12, 18]));
    model.setOperandValue(op33, new Int32Array([0, 0, 0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-8', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145];
    let op43_expect = [157, 163, 169, 193, 208, 223, 229, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type4 = {type: nn.INT32};
    let type73 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 1.0, zeroPoint: 127};
    let type73_length = product(type73.dimensions);
    let type74 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type74_length = product(type74.dimensions);
    let type75 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type75_length = product(type75.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type73);
    let op23 = operandIndex++;
    model.addOperand(type74);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 1.005])});
    let op33 = operandIndex++;
    model.addOperand(type75);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type73);

    model.setOperandValue(op23, new Int8Array([2, 8, 14, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op33, new Int32Array([0, 0, 0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type73_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type73_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-9', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op23_value = [2, 8, 14, 2, 5, 8, 6, 12, 18];
    let op33_value = [0, 0, 0];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);
    let type71 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type71_length = product(type71.dimensions);
    let type72 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type72_length = product(type72.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type71);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 0.5])});
    let op33 = operandIndex++;
    model.addOperand(type72);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Int8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-10', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145];
    let op23_value = [2, 8, 14, 2, 5, 8, 3, 6, 9];
    let op33_value = [0, 0, 0];
    let op43_expect = [157, 163, 169, 193, 208, 223, 229, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type4 = {type: nn.INT32};
    let type73 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 1.0, zeroPoint: 127};
    let type73_length = product(type73.dimensions);
    let type74 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type74_length = product(type74.dimensions);
    let type75 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.0, zeroPoint: 0};
    let type75_length = product(type75.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type73);
    let op23 = operandIndex++;
    model.addOperand(type74);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 1.005])});
    let op33 = operandIndex++;
    model.addOperand(type75);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type73);

    model.setOperandValue(op23, new Int8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type73_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type73_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });


  it('check result for Depthwise conv2d per channel example-1', async function() {
    // For 'Depthwise conv2d per channel' example: examples_same
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 16, 4, 32, 4, 64, 4, 128];
    let op4_expect = [8, 48];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 0};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 1.0, zeroPoint: 0};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.5, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type2);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Int8Array([2, 4, 2, 0, 2, 2, 2, 0]));
    model.setOperandValue(op3, new Int32Array([0, 0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-2', async function() {
    // For 'Depthwise conv2d per channel' example: examples_same_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 16, 4, 32, 4, 64, 4, 128];
    let op2_value = [2, 4, 2, 0, 2, 2, 2, 0];
    let op3_value = [0, 0];
    let op4_expect = [8, 48];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 0};
    let type0_length = product(type0.dimensions);
    let type11 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type11_length = product(type11.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 1.0, zeroPoint: 0};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type11);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.5, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type2);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-3', async function() {
    // For 'Depthwise conv2d per channel' example: examples_different
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op41_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type6_length = product(type6.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type6);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op31 = operandIndex++;
    model.addOperand(type7);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op21, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op31, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([1]));
    model.setOperandValue(param14, new Int32Array([2]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12, param13, param14, param15], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type8_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-4', async function() {
    // For 'Depthwise conv2d per channel' example: examples_different_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op21_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let op31_value = [4, 4, 4, 4];
    let op41_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type12 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type12_length = product(type12.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type12);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op31 = operandIndex++;
    model.addOperand(type7);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([1]));
    model.setOperandValue(param14, new Int32Array([2]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12, param13, param14, param15], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type8_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-5', async function() {
    // For 'Depthwise conv2d per channel' example: examples_layout_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op42_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type10 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type10_length = product(type10.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type5);
    let op22 = operandIndex++;
    model.addOperand(type10);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op32 = operandIndex++;
    model.addOperand(type7);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op22, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op32, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([1]));
    model.setOperandValue(param21, new Int32Array([1]));
    model.setOperandValue(param22, new Int32Array([2]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param16, param17, param18, param19, param20, param21, param22, param23], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type8_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-6', async function() {
    // For 'Depthwise conv2d per channel' example: examples_layout_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op22_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let op32_value = [4, 4, 4, 4];
    let op42_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type13 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type13_length = product(type13.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type5);
    let op22 = operandIndex++;
    model.addOperand(type13);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op32 = operandIndex++;
    model.addOperand(type7);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op22, new Int8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([1]));
    model.setOperandValue(param21, new Int32Array([1]));
    model.setOperandValue(param22, new Int32Array([2]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param16, param17, param18, param19, param20, param21, param22, param23], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type8_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });


  it('check result for Depthwise conv2d quant8 signed example-1', async function() {
    // For 'Depthwise conv2d quant8 signed' example: examples_same
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-124, -112, -124, -96, -124, -64, -124, 0];
    let op45_expect = [-120, -80];

    let type10 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: -128};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type12_length = product(type12.dimensions);
    let type13 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 1, 1, 2], scale: 1.0, zeroPoint: -128};
    let type13_length = product(type13.dimensions);
    let type4 = {type: nn.INT32};

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type11);
    model.setOperandSymmPerChannelQuantParams(op25, {channelDim: 3, scales: new Float32Array([0.5, 0.5])});
    let op35 = operandIndex++;
    model.addOperand(type12);
    let param41 = operandIndex++;
    model.addOperand(type4);
    let param42 = operandIndex++;
    model.addOperand(type4);
    let param43 = operandIndex++;
    model.addOperand(type4);
    let param44 = operandIndex++;
    model.addOperand(type4);
    let param45 = operandIndex++;
    model.addOperand(type4);
    let param46 = operandIndex++;
    model.addOperand(type4);
    let param47 = operandIndex++;
    model.addOperand(type4);
    let param48 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type13);

    model.setOperandValue(op25, new Int8Array([2, 4, 2, 0, 2, 2, 2, 0]));
    model.setOperandValue(op35, new Int32Array([0, 0]));
    model.setOperandValue(param41, new Int32Array([0]));
    model.setOperandValue(param42, new Int32Array([0]));
    model.setOperandValue(param43, new Int32Array([0]));
    model.setOperandValue(param44, new Int32Array([0]));
    model.setOperandValue(param45, new Int32Array([1]));
    model.setOperandValue(param46, new Int32Array([1]));
    model.setOperandValue(param47, new Int32Array([1]));
    model.setOperandValue(param48, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op15, op25, op35, param41, param42, param43, param44, param45, param46, param47, param48], [op45]);

    model.identifyInputsAndOutputs([op15], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Int8Array(op15_value);
    execution.setInput(0, op15_input);
    let op45_output = new Int8Array(type13_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type13_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });

  it('check result for Depthwise conv2d quant8 signed example-2', async function() {
    // For 'Depthwise conv2d quant8 signed' example: examples_different
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op16_value = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
    let op46_expect = [4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3];

    let type14 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type14_length = product(type14.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type15_length = product(type15.dimensions);
    let type16 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 0};
    let type17_length = product(type17.dimensions);
    let type4 = {type: nn.INT32};

    let op16 = operandIndex++;
    model.addOperand(type14);
    let op26 = operandIndex++;
    model.addOperand(type15);
    model.setOperandSymmPerChannelQuantParams(op26, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op36 = operandIndex++;
    model.addOperand(type16);
    let param49 = operandIndex++;
    model.addOperand(type4);
    let param50 = operandIndex++;
    model.addOperand(type4);
    let param51 = operandIndex++;
    model.addOperand(type4);
    let param52 = operandIndex++;
    model.addOperand(type4);
    let param53 = operandIndex++;
    model.addOperand(type4);
    let param54 = operandIndex++;
    model.addOperand(type4);
    let param55 = operandIndex++;
    model.addOperand(type4);
    let param56 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(op26, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op36, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param49, new Int32Array([0]));
    model.setOperandValue(param50, new Int32Array([0]));
    model.setOperandValue(param51, new Int32Array([0]));
    model.setOperandValue(param52, new Int32Array([0]));
    model.setOperandValue(param53, new Int32Array([1]));
    model.setOperandValue(param54, new Int32Array([1]));
    model.setOperandValue(param55, new Int32Array([2]));
    model.setOperandValue(param56, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op16, op26, op36, param49, param50, param51, param52, param53, param54, param55, param56], [op46]);

    model.identifyInputsAndOutputs([op16], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op16_input = new Int8Array(op16_value);
    execution.setInput(0, op16_input);
    let op46_output = new Int8Array(type17_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });

  it('check result for Depthwise conv2d quant8 signed example-3', async function() {
    // For 'Depthwise conv2d quant8 signed' example: examples_layout_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op17_value = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
    let op47_expect = [4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3];

    let type14 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type14_length = product(type14.dimensions);
    let type16 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 0};
    let type17_length = product(type17.dimensions);
    let type18 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type18_length = product(type18.dimensions);
    let type4 = {type: nn.INT32};

    let op17 = operandIndex++;
    model.addOperand(type14);
    let op27 = operandIndex++;
    model.addOperand(type18);
    model.setOperandSymmPerChannelQuantParams(op27, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op37 = operandIndex++;
    model.addOperand(type16);
    let param57 = operandIndex++;
    model.addOperand(type4);
    let param58 = operandIndex++;
    model.addOperand(type4);
    let param59 = operandIndex++;
    model.addOperand(type4);
    let param60 = operandIndex++;
    model.addOperand(type4);
    let param61 = operandIndex++;
    model.addOperand(type4);
    let param62 = operandIndex++;
    model.addOperand(type4);
    let param63 = operandIndex++;
    model.addOperand(type4);
    let param64 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(op27, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op37, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param57, new Int32Array([0]));
    model.setOperandValue(param58, new Int32Array([0]));
    model.setOperandValue(param59, new Int32Array([0]));
    model.setOperandValue(param60, new Int32Array([0]));
    model.setOperandValue(param61, new Int32Array([1]));
    model.setOperandValue(param62, new Int32Array([1]));
    model.setOperandValue(param63, new Int32Array([2]));
    model.setOperandValue(param64, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op17, op27, op37, param57, param58, param59, param60, param61, param62, param63, param64], [op47]);

    model.identifyInputsAndOutputs([op17], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op17_input = new Int8Array(op17_value);
    execution.setInput(0, op17_input);
    let op47_output = new Int8Array(type17_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });


  it('check result for Depthwise conv2d v1_2 example-1', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op4_expect = [110, 30, 72, 106, 110, 30, 74, 109, 110, 30, 78, 115, 110, 30, 80, 118];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type19 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type19_length = product(type19.dimensions);
    let type20 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type20_length = product(type20.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.1, zeroPoint: 0};
    let type21_length = product(type21.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type19);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type20);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type21);

    model.setOperandValue(op2, new Int8Array([25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0]));
    model.setOperandValue(op3, new Int32Array([200, 800, 600, 1600]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type21_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type21_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-2', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op4_expect = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type22 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type23_length = product(type23.dimensions);
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.0001, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type22);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type23);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type24);

    model.setOperandValue(op2, new Int8Array([25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0]));
    model.setOperandValue(op3, new Int32Array([200, 800, 600, 1600]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type24_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type24_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-3', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op2_value = [25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0];
    let op3_value = [200, 800, 600, 1600];
    let op4_expect = [110, 30, 72, 106, 110, 30, 74, 109, 110, 30, 78, 115, 110, 30, 80, 118];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type19 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type19_length = product(type19.dimensions);
    let type20 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type20_length = product(type20.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.1, zeroPoint: 0};
    let type21_length = product(type21.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type19);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type20);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type21);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type21_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type21_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-4', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op2_value = [25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0];
    let op3_value = [200, 800, 600, 1600];
    let op4_expect = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type22 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type23_length = product(type23.dimensions);
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.0001, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type22);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type23);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type24);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type24_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type24_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-5', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_channelQuant8_3
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [130, 132, 142, 144, 134, 136, 146, 148, 138, 140, 150, 152];
    let op41_expect = [171, 66, 199, 80, 191, 74, 227, 96];

    let type36 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.5, zeroPoint: 128};
    let type36_length = product(type36.dimensions);
    let type39 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 1, 4], scale: 1.0, zeroPoint: 100};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type36);
    let op21 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([0.5, 0.25, 0.5, 0.25])});
    let op31 = operandIndex++;
    model.addOperand(type41);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type39);

    model.setOperandValue(op21, new Int8Array([2, 8, 6, 16, -18, 40, -22, 48, 10, 24, 14, 32, 26, -56, 30, -64]));
    model.setOperandValue(op31, new Int32Array([4, 16, 12, 32]));
    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type39_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type39_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-6', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8_3
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [130, 132, 142, 144, 134, 136, 146, 148, 138, 140, 150, 152];
    let op21_value = [2, 8, 6, 16, -18, 40, -22, 48, 10, 24, 14, 32, 26, -56, 30, -64];
    let op31_value = [4, 16, 12, 32];
    let op41_expect = [171, 66, 199, 80, 191, 74, 227, 96];

    let type36 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.5, zeroPoint: 128};
    let type36_length = product(type36.dimensions);
    let type39 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 1, 4], scale: 1.0, zeroPoint: 100};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type36);
    let op21 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([0.5, 0.25, 0.5, 0.25])});
    let op31 = operandIndex++;
    model.addOperand(type41);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type39);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type39_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type39_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-7', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [148, 170, 148, 172, 148, 174, 148, 176];
    let op42_expect = [183, 251];

    let type4 = {type: nn.INT32};
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 2.0, zeroPoint: 128};
    let type54_length = product(type54.dimensions);
    let type55 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 128};
    let type55_length = product(type55.dimensions);
    let type56 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type56_length = product(type56.dimensions);
    let type57 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type57_length = product(type57.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type55);
    let op22 = operandIndex++;
    model.addOperand(type56);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([0.125, 0.25])});
    let op32 = operandIndex++;
    model.addOperand(type57);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op22, new Int8Array([2, 0, 2, 4, 2, 0, 2, 4]));
    model.setOperandValue(op32, new Int32Array([1600, 1600]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type54_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-8', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [148, 170, 148, 172, 148, 174, 148, 176];
    let op22_value = [2, 0, 2, 4, 2, 0, 2, 4];
    let op32_value = [1600, 1600];
    let op42_expect = [183, 251];

    let type4 = {type: nn.INT32};
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 2.0, zeroPoint: 128};
    let type54_length = product(type54.dimensions);
    let type55 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 128};
    let type55_length = product(type55.dimensions);
    let type56 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type56_length = product(type56.dimensions);
    let type57 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type57_length = product(type57.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type55);
    let op22 = operandIndex++;
    model.addOperand(type56);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([0.125, 0.25])});
    let op32 = operandIndex++;
    model.addOperand(type57);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op22, new Int8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type54_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-9', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [148, 170, 148, 128, 148, 172, 168, 128, 148, 174, 188, 128, 148, 176, 208, 128];
    let op43_expect = [120, 141, 220, 180];

    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 4], scale: 50.0, zeroPoint: 0};
    let type64_length = product(type64.dimensions);
    let type65 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type65_length = product(type65.dimensions);
    let type66 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type66_length = product(type66.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type37);
    let op23 = operandIndex++;
    model.addOperand(type65);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 3, scales: new Float32Array([1.0, 2.0, 1.0, 1.0])});
    let op33 = operandIndex++;
    model.addOperand(type66);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type64);

    model.setOperandValue(op23, new Int8Array([0, 0, 10, 50, 0, 0, 20, 50, 0, 0, 30, 50, 0, 0, 40, 50]));
    model.setOperandValue(op33, new Int32Array([12000, 7000, 16000, 18000]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type64_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type64_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-10', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [148, 170, 148, 128, 148, 172, 168, 128, 148, 174, 188, 128, 148, 176, 208, 128];
    let op23_value = [0, 0, 10, 50, 0, 0, 20, 50, 0, 0, 30, 50, 0, 0, 40, 50];
    let op33_value = [12000, 7000, 16000, 18000];
    let op43_expect = [120, 141, 220, 180];

    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 4], scale: 50.0, zeroPoint: 0};
    let type64_length = product(type64.dimensions);
    let type65 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type65_length = product(type65.dimensions);
    let type66 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type66_length = product(type66.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type37);
    let op23 = operandIndex++;
    model.addOperand(type65);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 3, scales: new Float32Array([1.0, 2.0, 1.0, 1.0])});
    let op33 = operandIndex++;
    model.addOperand(type66);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type64);

    model.setOperandValue(op23, new Int8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type64_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type64_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });
});
