// Generated file (from: depthwise_conv2d_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.5, 0.25, 0.5, 0.25])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.5, 0.25, 0.5, 0.25])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.125, 0.25])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([0.125, 0.25])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([1.0, 2.0, 1.0, 1.0])});
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
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 3, scales: new Float32Array([1.0, 2.0, 1.0, 1.0])});
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
