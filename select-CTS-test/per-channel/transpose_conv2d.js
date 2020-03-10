// Generated file (from: transpose_conv2d.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Transpose conv2d example-1', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_none_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [79, 80, 83, 84, 91, 96, 89, 92, 97, 100, 91, 92, 95, 96, 127, 132, 113, 116, 121, 124, 109, 116, 125, 132, 201, 220, 161, 172, 185, 196, 119, 124, 131, 136, 199, 212, 149, 156, 165, 172, 155, 160, 167, 172, 255, 255, 197, 204, 213, 220];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type36_length = product(type36.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-2', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_none_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [79, 80, 83, 84, 91, 96, 89, 92, 97, 100, 91, 92, 95, 96, 127, 132, 113, 116, 121, 124, 109, 116, 125, 132, 201, 220, 161, 172, 185, 196, 119, 124, 131, 136, 199, 212, 149, 156, 165, 172, 155, 160, 167, 172, 255, 255, 197, 204, 213, 220];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type38 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type38_length = product(type38.dimensions);
    let type39 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type38);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type39);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-3', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_none_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [75, 80, 95, 100, 135, 160, 125, 140, 165, 180, 135, 140, 155, 160, 255, 255, 245, 255, 255, 255, 225, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type41);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-4', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_none_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [75, 80, 95, 100, 135, 160, 125, 140, 165, 180, 135, 140, 155, 160, 255, 255, 245, 255, 255, 255, 225, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type42 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type42_length = product(type42.dimensions);
    let type43 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type43_length = product(type43.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type42);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type43);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-5', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [80, 80, 83, 84, 91, 96, 89, 92, 97, 100, 91, 92, 95, 96, 127, 132, 113, 116, 121, 124, 109, 116, 125, 132, 201, 220, 161, 172, 185, 196, 119, 124, 131, 136, 199, 212, 149, 156, 165, 172, 155, 160, 167, 172, 255, 255, 197, 204, 213, 220];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type36_length = product(type36.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-6', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [80, 80, 83, 84, 91, 96, 89, 92, 97, 100, 91, 92, 95, 96, 127, 132, 113, 116, 121, 124, 109, 116, 125, 132, 201, 220, 161, 172, 185, 196, 119, 124, 131, 136, 199, 212, 149, 156, 165, 172, 155, 160, 167, 172, 255, 255, 197, 204, 213, 220];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type50 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type50_length = product(type50.dimensions);
    let type51 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type51_length = product(type51.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type50);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type51);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-7', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [80, 80, 95, 100, 135, 160, 125, 140, 165, 180, 135, 140, 155, 160, 255, 255, 245, 255, 255, 255, 225, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type41);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-8', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [80, 80, 95, 100, 135, 160, 125, 140, 165, 180, 135, 140, 155, 160, 255, 255, 245, 255, 255, 255, 225, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type52 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type52_length = product(type52.dimensions);
    let type53 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type53_length = product(type53.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type52);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type53);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([1]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-9', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu1_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [79, 80, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type36_length = product(type36.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-10', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu1_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [79, 80, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type54 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type54_length = product(type54.dimensions);
    let type55 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type55_length = product(type55.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type54);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type55);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-11', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu1_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [75, 80, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type41);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-12', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu1_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [75, 80, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type56 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type56_length = product(type56.dimensions);
    let type57 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type57_length = product(type57.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type56);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type57);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([2]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-13', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu6_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [80, 80, 83, 84, 91, 92, 89, 92, 92, 92, 91, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type36_length = product(type36.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-14', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu6_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [80, 80, 83, 84, 91, 92, 89, 92, 92, 92, 91, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92];

    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.5, zeroPoint: 80};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type58 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type58_length = product(type58.dimensions);
    let type59 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type59_length = product(type59.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type58);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type59);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type37);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type37_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type37_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-15', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu6_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op4_expect = [80, 80, 95, 100, 135, 140, 125, 140, 140, 140, 135, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type41);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array([4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36]));
    model.setOperandValue(op3, new Int32Array([-24, -16]));
    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-16', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_relu6_channelQuant8_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 108, 112, 116];
    let op2_value = [4, 12, 20, 28, 36, 44, 52, 60, 68, 4, 8, 12, 16, 20, 24, 28, 32, 36];
    let op3_value = [-24, -16];
    let op4_expect = [80, 80, 95, 100, 135, 140, 125, 140, 140, 140, 135, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140];

    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 2], scale: 0.1, zeroPoint: 80};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.25, zeroPoint: 100};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type60 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 3, 1]};
    let type60_length = product(type60.dimensions);
    let type61 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type61_length = product(type61.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type34);
    let op2 = operandIndex++;
    model.addOperand(type60);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type61);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(shape, new Int32Array([1, 5, 5, 2]));
    model.setOperandValue(param, new Int32Array([2]));
    model.setOperandValue(param1, new Int32Array([2]));
    model.setOperandValue(param2, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

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

  it('check result for Transpose conv2d example-17', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [150, 250];
    let op41_expect = [75, 90, 225, 125, 120, 75, 225, 200, 50, 60, 75, 50];

    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type88 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 2, 1], scale: 2.0, zeroPoint: 0};
    let type88_length = product(type88.dimensions);
    let type91 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 20.0, zeroPoint: 50};
    let type91_length = product(type91.dimensions);
    let type92 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type92_length = product(type92.dimensions);
    let type93 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type93_length = product(type93.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type88);
    let op21 = operandIndex++;
    model.addOperand(type92);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25])});
    let op31 = operandIndex++;
    model.addOperand(type93);
    let shape1 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type5);
    let param4 = operandIndex++;
    model.addOperand(type5);
    let param5 = operandIndex++;
    model.addOperand(type5);
    let param6 = operandIndex++;
    model.addOperand(type5);
    let op41 = operandIndex++;
    model.addOperand(type91);

    model.setOperandValue(op21, new Int8Array([36, 20, 24, 36, 32, 20, 12, 4, 16]));
    model.setOperandValue(op31, new Int32Array([-2000]));
    model.setOperandValue(shape1, new Int32Array([1, 3, 4, 1]));
    model.setOperandValue(param3, new Int32Array([1]));
    model.setOperandValue(param4, new Int32Array([3]));
    model.setOperandValue(param5, new Int32Array([3]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op11, op21, op31, shape1, param3, param4, param5, param6], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type91_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type91_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Transpose conv2d example-18', async function() {
    // For 'Transpose conv2d' example: examples_nhwc_channelQuant8_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [150, 250];
    let op21_value = [36, 20, 24, 36, 32, 20, 12, 4, 16];
    let op31_value = [-2000];
    let op41_expect = [75, 90, 225, 125, 120, 75, 225, 200, 50, 60, 75, 50];

    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};
    let type88 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 2, 1], scale: 2.0, zeroPoint: 0};
    let type88_length = product(type88.dimensions);
    let type91 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 20.0, zeroPoint: 50};
    let type91_length = product(type91.dimensions);
    let type94 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type94_length = product(type94.dimensions);
    let type95 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0, zeroPoint: 0};
    let type95_length = product(type95.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type88);
    let op21 = operandIndex++;
    model.addOperand(type94);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim=0, scales=new Float32Array([0.25])});
    let op31 = operandIndex++;
    model.addOperand(type95);
    let shape1 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type5);
    let param4 = operandIndex++;
    model.addOperand(type5);
    let param5 = operandIndex++;
    model.addOperand(type5);
    let param6 = operandIndex++;
    model.addOperand(type5);
    let op41 = operandIndex++;
    model.addOperand(type91);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(shape1, new Int32Array([1, 3, 4, 1]));
    model.setOperandValue(param3, new Int32Array([1]));
    model.setOperandValue(param4, new Int32Array([3]));
    model.setOperandValue(param5, new Int32Array([3]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op11, op21, op31, shape1, param3, param4, param5, param6], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type91_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type91_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });
});
