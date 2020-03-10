// Generated file (from: grouped_conv2d.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Grouped conv2d example-1', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_none_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [146, 79, 146, 95, 142, 89, 134, 61];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type17_length = product(type17.dimensions);
    let type18 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type17);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type18);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-2', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_none_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [146, 79, 146, 95, 142, 89, 134, 61];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type19 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type19_length = product(type19.dimensions);
    let type20 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type20_length = product(type20.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type19);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-3', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_none_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [255, 75, 255, 155, 255, 125, 255, 0];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type21_length = product(type21.dimensions);
    let type22 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type21);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type22);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-4', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_none_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [255, 75, 255, 155, 255, 125, 255, 0];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type24 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type24_length = product(type24.dimensions);
    let type25 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type25_length = product(type25.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type24);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type25);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-5', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [146, 80, 146, 95, 142, 89, 134, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type17_length = product(type17.dimensions);
    let type18 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type17);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type18);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-6', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [146, 80, 146, 95, 142, 89, 134, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type32 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type32);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type33);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-7', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [255, 80, 255, 155, 255, 125, 255, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type21_length = product(type21.dimensions);
    let type22 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type21);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type22);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-8', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [255, 80, 255, 155, 255, 125, 255, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type34_length = product(type34.dimensions);
    let type35 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type35_length = product(type35.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type34);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type35);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-9', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu1_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [82, 79, 82, 82, 82, 82, 82, 78];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type17_length = product(type17.dimensions);
    let type18 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type17);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type18);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-10', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu1_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [82, 79, 82, 82, 82, 82, 82, 78];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type36 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type36_length = product(type36.dimensions);
    let type37 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type36);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type37);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-11', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu1_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [90, 75, 90, 90, 90, 90, 90, 70];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type21_length = product(type21.dimensions);
    let type22 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type21);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type22);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-12', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu1_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [90, 75, 90, 90, 90, 90, 90, 70];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type38 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type38_length = product(type38.dimensions);
    let type39 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type38);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type39);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-13', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu6_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [92, 80, 92, 92, 92, 89, 92, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type17_length = product(type17.dimensions);
    let type18 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type17);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type18);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-14', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu6_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [92, 80, 92, 92, 92, 89, 92, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 80};
    let type15_length = product(type15.dimensions);
    let type4 = {type: nn.INT32};
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type41);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type15_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-15', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu6_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op4_expect = [140, 80, 140, 140, 140, 125, 140, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type21_length = product(type21.dimensions);
    let type22 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type21);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type22);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array([4, 8, 8, 4, 8, 6, 4, 2]));
    model.setOperandValue(op3, new Int32Array([160, -268]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-16', async function() {
    // For 'Grouped conv2d' example: examples_nhwc_relu6_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116, 120, 124, 124, 120, 116, 112, 108, 104, 108, 112, 112, 112, 112, 112];
    let op2_value = [4, 8, 8, 4, 8, 6, 4, 2];
    let op3_value = [160, -268];
    let op4_expect = [140, 80, 140, 140, 140, 125, 140, 80];

    let type12 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.25, zeroPoint: 100};
    let type12_length = product(type12.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.1, zeroPoint: 80};
    let type23_length = product(type23.dimensions);
    let type4 = {type: nn.INT32};
    let type42 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 2, 1]};
    let type42_length = product(type42.dimensions);
    let type43 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type43_length = product(type43.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type12);
    let op2 = operandIndex++;
    model.addOperand(type42);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type43);
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
    let act = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.GROUPED_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type23_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-17', async function() {
    // For 'Grouped conv2d' example: examples_large_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [132, 136, 140, 144, 144, 140, 136, 132, 136, 140, 140, 140];
    let op41_expect = [157, 13, 248, 84, 161, 16, 237, 99, 154, 9, 176, 69];

    let type4 = {type: nn.INT32};
    let type63 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.25, zeroPoint: 128};
    let type63_length = product(type63.dimensions);
    let type66 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 10.0, zeroPoint: 100};
    let type66_length = product(type66.dimensions);
    let type67 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 3, 1]};
    let type67_length = product(type67.dimensions);
    let type68 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type68_length = product(type68.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type63);
    let op21 = operandIndex++;
    model.addOperand(type67);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([2.0, 2.5])});
    let op31 = operandIndex++;
    model.addOperand(type68);
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
    let op41 = operandIndex++;
    model.addOperand(type66);

    model.setOperandValue(op21, new Int8Array([50, 10, 0, 100, 5, 1, 80, 12, 0, 40, 8, 1]));
    model.setOperandValue(op31, new Int32Array([1000, -1600]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([2]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op11, op21, op31, param7, param8, param9, param10, param11], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type66_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type66_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-18', async function() {
    // For 'Grouped conv2d' example: examples_large_nhwc_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [132, 136, 140, 144, 144, 140, 136, 132, 136, 140, 140, 140];
    let op21_value = [50, 10, 0, 100, 5, 1, 80, 12, 0, 40, 8, 1];
    let op31_value = [1000, -1600];
    let op41_expect = [157, 13, 248, 84, 161, 16, 237, 99, 154, 9, 176, 69];

    let type4 = {type: nn.INT32};
    let type63 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.25, zeroPoint: 128};
    let type63_length = product(type63.dimensions);
    let type66 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 10.0, zeroPoint: 100};
    let type66_length = product(type66.dimensions);
    let type69 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 2, 3, 1]};
    let type69_length = product(type69.dimensions);
    let type70 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type63);
    let op21 = operandIndex++;
    model.addOperand(type69);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([2.0, 2.5])});
    let op31 = operandIndex++;
    model.addOperand(type70);
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
    let op41 = operandIndex++;
    model.addOperand(type66);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([2]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op11, op21, op31, param7, param8, param9, param10, param11], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type66_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type66_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-19', async function() {
    // For 'Grouped conv2d' example: examples_channel_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [2, 4, 6, 8, 110, 8, 6, 4, 2, 10, 8, 6, 4, 22, 4, 6, 8, 10, 4, 6, 4, 6, 44, 6, 4, 6, 4, 2, 0, 4, 2, 66, 2, 4, 0, 2];
    let op42_expect = [72, 52, 168, 229, 109, 34, 76, 57, 96, 85, 127, 38, 72, 54, 116, 124, 111, 34, 68, 51, 127, 145, 96, 32];

    let type4 = {type: nn.INT32};
    let type80 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 9], scale: 0.5, zeroPoint: 0};
    let type80_length = product(type80.dimensions);
    let type83 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 6], scale: 2.0, zeroPoint: 60};
    let type83_length = product(type83.dimensions);
    let type84 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [6, 1, 1, 3]};
    let type84_length = product(type84.dimensions);
    let type85 = {type: nn.TENSOR_INT32, dimensions: [6], scale: 0.0, zeroPoint: 0};
    let type85_length = product(type85.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type80);
    let op22 = operandIndex++;
    model.addOperand(type84);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.3, 0.25, 0.3, 0.25, 0.3])});
    let op32 = operandIndex++;
    model.addOperand(type85);
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
    let op42 = operandIndex++;
    model.addOperand(type83);

    model.setOperandValue(op22, new Int8Array([4, 8, 12, 7, 3, 0, 8, 12, 12, 20, 20, 20, 36, 32, 20, 7, 3, 3]));
    model.setOperandValue(op32, new Int32Array([80, -133, 240, -267, 400, -400]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([1]));
    model.setOperandValue(param14, new Int32Array([1]));
    model.setOperandValue(param15, new Int32Array([3]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op12, op22, op32, param12, param13, param14, param15, param16], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type83_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type83_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Grouped conv2d example-20', async function() {
    // For 'Grouped conv2d' example: examples_channel_nhwc_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [2, 4, 6, 8, 110, 8, 6, 4, 2, 10, 8, 6, 4, 22, 4, 6, 8, 10, 4, 6, 4, 6, 44, 6, 4, 6, 4, 2, 0, 4, 2, 66, 2, 4, 0, 2];
    let op22_value = [4, 8, 12, 7, 3, 0, 8, 12, 12, 20, 20, 20, 36, 32, 20, 7, 3, 3];
    let op32_value = [80, -133, 240, -267, 400, -400];
    let op42_expect = [72, 52, 168, 229, 109, 34, 76, 57, 96, 85, 127, 38, 72, 54, 116, 124, 111, 34, 68, 51, 127, 145, 96, 32];

    let type4 = {type: nn.INT32};
    let type80 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 9], scale: 0.5, zeroPoint: 0};
    let type80_length = product(type80.dimensions);
    let type83 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 6], scale: 2.0, zeroPoint: 60};
    let type83_length = product(type83.dimensions);
    let type86 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [6, 1, 1, 3]};
    let type86_length = product(type86.dimensions);
    let type87 = {type: nn.TENSOR_INT32, dimensions: [6], scale: 0.0, zeroPoint: 0};
    let type87_length = product(type87.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type80);
    let op22 = operandIndex++;
    model.addOperand(type86);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.3, 0.25, 0.3, 0.25, 0.3])});
    let op32 = operandIndex++;
    model.addOperand(type87);
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
    let op42 = operandIndex++;
    model.addOperand(type83);

    model.setOperandValue(op22, new Int8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([1]));
    model.setOperandValue(param14, new Int32Array([1]));
    model.setOperandValue(param15, new Int32Array([3]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.addOperation(nn.GROUPED_CONV_2D, [op12, op22, op32, param12, param13, param14, param15, param16], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type83_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type83_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });
});
