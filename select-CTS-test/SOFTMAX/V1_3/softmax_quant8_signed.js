// Generated file (from: softmax_quant8_signed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Softmax quant8 signed example-1', async function() {
    // For 'Softmax quant8 signed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [-127, -126, -118, -108];
    let output_expect = [-64, -64, -64, -64];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 4], scale: 0.5, zeroPoint: -128};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.FLOAT32};
    let type2 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 4], scale: 0.00390625, zeroPoint: -128};
    let type2_length = product(type2.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(beta, new Float32Array([1e-05]));
    model.addOperation(nn.SOFTMAX, [input, beta], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Int8Array(input_value);
    execution.setInput(0, input_input);
    let output_output = new Int8Array(type2_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(output_output[i], output_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-2', async function() {
    // For 'Softmax quant8 signed' example: examples_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value = [-127, -126, -125, -124, -123, 127, 126, 125, 124, 123];
    let output1_expect = [-113, -104, -88, -61, -18, -18, -61, -88, -104, -113];

    let type1 = {type: nn.FLOAT32};
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.5, zeroPoint: -128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.00390625, zeroPoint: -128};
    let type4_length = product(type4.dimensions);

    let input1 = operandIndex++;
    model.addOperand(type3);
    let beta1 = operandIndex++;
    model.addOperand(type1);
    let output1 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(beta1, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [input1, beta1], [output1]);

    model.identifyInputsAndOutputs([input1], [output1]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1_input = new Int8Array(input1_value);
    execution.setInput(0, input1_input);
    let output1_output = new Int8Array(type4_length);
    execution.setOutput(0, output1_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(output1_output[i], output1_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-3', async function() {
    // For 'Softmax quant8 signed' example: examples_float32
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68];
    let op2_expect = [37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128];

    let type1 = {type: nn.FLOAT32};
    let type16 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 0};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: -128};
    let type17_length = product(type17.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type16);
    let param = operandIndex++;
    model.addOperand(type1);
    let op2 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [op1, param], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type17_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-4', async function() {
    // For 'Softmax quant8 signed' example: examples_float32_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type16 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 0};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: -128};
    let type17_length = product(type17.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type16);
    let param1 = operandIndex++;
    model.addOperand(type1);
    let op2 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param1, new Float32Array([1e-06]));
    model.addOperation(nn.SOFTMAX, [op1, param1], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type17_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-5', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis0
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 68, -4, 68, -4, 68, -4, 64, -8, 64, -8, 64, -8, 64, -8, 60, -12, 60, -12, 60, -12, 60, -12, 56, -16, 56, -16, 56, -16, 56, -16, 4, -68, 4, -68, 4, -68, 4, -68];
    let op2_expect = [37, 37, 37, 37, 37, 37, 37, 37, -67, -67, -67, -67, -67, -67, -67, -67, -106, -106, -106, -106, -106, -106, -106, -106, -120, -120, -120, -120, -120, -120, -120, -120, -128, -128, -128, -128, -128, -128, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type22 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.25, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type23_length = product(type23.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type22);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([0]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type23_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-6', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis0_neg
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 68, -4, 68, -4, 68, -4, 64, -8, 64, -8, 64, -8, 64, -8, 60, -12, 60, -12, 60, -12, 60, -12, 56, -16, 56, -16, 56, -16, 56, -16, 4, -68, 4, -68, 4, -68, 4, -68];
    let op2_expect = [37, 37, 37, 37, 37, 37, 37, 37, -67, -67, -67, -67, -67, -67, -67, -67, -106, -106, -106, -106, -106, -106, -106, -106, -120, -120, -120, -120, -120, -120, -120, -120, -128, -128, -128, -128, -128, -128, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type22 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.25, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type23_length = product(type23.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type22);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([-4]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type23_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-7', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis1
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 68, -4, 64, -8, 64, -8, 60, -12, 60, -12, 56, -16, 56, -16, 4, -68, 4, -68, 68, -4, 68, -4, 64, -8, 64, -8, 60, -12, 60, -12, 56, -16, 56, -16, 4, -68, 4, -68];
    let op2_expect = [37, 37, 37, 37, -67, -67, -67, -67, -106, -106, -106, -106, -120, -120, -120, -120, -128, -128, -128, -128, 37, 37, 37, 37, -67, -67, -67, -67, -106, -106, -106, -106, -120, -120, -120, -120, -128, -128, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.25, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type25 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type25_length = product(type25.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type24);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type25);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([1]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type25_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type25_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-8', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis1_neg
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 68, -4, 64, -8, 64, -8, 60, -12, 60, -12, 56, -16, 56, -16, 4, -68, 4, -68, 68, -4, 68, -4, 64, -8, 64, -8, 60, -12, 60, -12, 56, -16, 56, -16, 4, -68, 4, -68];
    let op2_expect = [37, 37, 37, 37, -67, -67, -67, -67, -106, -106, -106, -106, -120, -120, -120, -120, -128, -128, -128, -128, 37, 37, 37, 37, -67, -67, -67, -67, -106, -106, -106, -106, -120, -120, -120, -120, -128, -128, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.25, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type25 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type25_length = product(type25.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type24);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type25);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([-3]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type25_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type25_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-9', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 64, -8, 60, -12, 56, -16, 4, -68, 68, -4, 64, -8, 60, -12, 56, -16, 4, -68, 68, -4, 64, -8, 60, -12, 56, -16, 4, -68, 68, -4, 64, -8, 60, -12, 56, -16, 4, -68];
    let op2_expect = [37, 37, -67, -67, -106, -106, -120, -120, -128, -128, 37, 37, -67, -67, -106, -106, -120, -120, -128, -128, 37, 37, -67, -67, -106, -106, -120, -120, -128, -128, 37, 37, -67, -67, -106, -106, -120, -120, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type26 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.25, zeroPoint: 0};
    let type26_length = product(type26.dimensions);
    let type27 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.00390625, zeroPoint: -128};
    let type27_length = product(type27.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type26);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type27);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([2]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type27_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type27_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-10', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis2_neg
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 64, -8, 60, -12, 56, -16, 4, -68, 68, -4, 64, -8, 60, -12, 56, -16, 4, -68, 68, -4, 64, -8, 60, -12, 56, -16, 4, -68, 68, -4, 64, -8, 60, -12, 56, -16, 4, -68];
    let op2_expect = [37, 37, -67, -67, -106, -106, -120, -120, -128, -128, 37, 37, -67, -67, -106, -106, -120, -120, -128, -128, 37, 37, -67, -67, -106, -106, -120, -120, -128, -128, 37, 37, -67, -67, -106, -106, -120, -120, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type26 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.25, zeroPoint: 0};
    let type26_length = product(type26.dimensions);
    let type27 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.00390625, zeroPoint: -128};
    let type27_length = product(type27.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type26);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type27);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([-2]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type27_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type27_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-11', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis3
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68];
    let op2_expect = [37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128];

    let type1 = {type: nn.FLOAT32};
    let type16 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 0};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: -128};
    let type17_length = product(type17.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type16);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([3]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type17_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-12', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis3_neg
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68, 68, 64, 60, 56, 4, -4, -8, -12, -16, -68];
    let op2_expect = [37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128, 37, -67, -106, -120, -128];

    let type1 = {type: nn.FLOAT32};
    let type16 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 0};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: -128};
    let type17_length = product(type17.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type16);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([-1]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type17_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-13', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis0
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 64, -8, 60, -12, 56, -16, 4, -68];
    let op2_expect = [37, 37, -67, -67, -106, -106, -120, -120, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type32 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.25, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.00390625, zeroPoint: -128};
    let type33_length = product(type33.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([0]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type33_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-14', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis0_neg
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, -4, 64, -8, 60, -12, 56, -16, 4, -68];
    let op2_expect = [37, 37, -67, -67, -106, -106, -120, -120, -128, -128];

    let type1 = {type: nn.FLOAT32};
    let type32 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.25, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.00390625, zeroPoint: -128};
    let type33_length = product(type33.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([-2]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type33_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-15', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis1
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, 64, 60, 56, 4, -4, -8, -12, -16, -68];
    let op2_expect = [37, -67, -106, -120, -128, 37, -67, -106, -120, -128];

    let type1 = {type: nn.FLOAT32};
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.25, zeroPoint: 0};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.00390625, zeroPoint: -128};
    let type4_length = product(type4.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([1]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type4_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-16', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis1_neg
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [68, 64, 60, 56, 4, -4, -8, -12, -16, -68];
    let op2_expect = [37, -67, -106, -120, -128, 37, -67, -106, -120, -128];

    let type1 = {type: nn.FLOAT32};
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.25, zeroPoint: 0};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.00390625, zeroPoint: -128};
    let type4_length = product(type4.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(param2, new Float32Array([1.0]));
    model.setOperandValue(axis, new Int32Array([-1]));
    model.addOperation(nn.SOFTMAX, [op1, param2, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type4_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-17', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis0_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 4, -4, 4, -4, 4, -4, 8, -8, 8, -8, 8, -8, 8, -8, 12, -12, 12, -12, 12, -12, 12, -12, 16, -16, 16, -16, 16, -16, 16, -16, 20, -20, 20, -20, 20, -20, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type22 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.25, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type23_length = product(type23.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type22);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([0]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type23_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-18', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis0_neg_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 4, -4, 4, -4, 4, -4, 8, -8, 8, -8, 8, -8, 8, -8, 12, -12, 12, -12, 12, -12, 12, -12, 16, -16, 16, -16, 16, -16, 16, -16, 20, -20, 20, -20, 20, -20, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type22 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.25, zeroPoint: 0};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type23_length = product(type23.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type22);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type23);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([-4]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type23_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type23_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-19', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis1_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 4, -4, 8, -8, 8, -8, 12, -12, 12, -12, 16, -16, 16, -16, 20, -20, 20, -20, 4, -4, 4, -4, 8, -8, 8, -8, 12, -12, 12, -12, 16, -16, 16, -16, 20, -20, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.25, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type25 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type25_length = product(type25.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type24);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type25);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([1]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type25_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type25_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-20', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis1_neg_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 4, -4, 8, -8, 8, -8, 12, -12, 12, -12, 16, -16, 16, -16, 20, -20, 20, -20, 4, -4, 4, -4, 8, -8, 8, -8, 12, -12, 12, -12, 16, -16, 16, -16, 20, -20, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.25, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type25 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5, 2, 2], scale: 0.00390625, zeroPoint: -128};
    let type25_length = product(type25.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type24);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type25);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([-3]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type25_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type25_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-21', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis2_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type26 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.25, zeroPoint: 0};
    let type26_length = product(type26.dimensions);
    let type27 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.00390625, zeroPoint: -128};
    let type27_length = product(type27.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type26);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type27);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([2]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type27_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type27_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-22', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis2_neg_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type26 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.25, zeroPoint: 0};
    let type26_length = product(type26.dimensions);
    let type27 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 5, 2], scale: 0.00390625, zeroPoint: -128};
    let type27_length = product(type27.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type26);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type27);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([-2]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type27_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type27_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-23', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis3_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type16 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 0};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: -128};
    let type17_length = product(type17.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type16);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([3]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type17_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-24', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim4_axis3_neg_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20, 4, 8, 12, 16, 20, -4, -8, -12, -16, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type16 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 0};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: -128};
    let type17_length = product(type17.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type16);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([-1]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type17_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-25', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis0_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 8, -8, 12, -12, 16, -16, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type32 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.25, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.00390625, zeroPoint: -128};
    let type33_length = product(type33.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([0]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type33_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-26', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis0_neg_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, -4, 8, -8, 12, -12, 16, -16, 20, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type32 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.25, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [5, 2], scale: 0.00390625, zeroPoint: -128};
    let type33_length = product(type33.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([-2]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type33_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-27', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis1_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 8, 12, 16, 20, -4, -8, -12, -16, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.25, zeroPoint: 0};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.00390625, zeroPoint: -128};
    let type4_length = product(type4.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([1]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type4_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax quant8 signed example-28', async function() {
    // For 'Softmax quant8 signed' example: examples_axis_float32_dim2_axis1_neg_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 8, 12, 16, 20, -4, -8, -12, -16, -20];
    let op2_expect = [-77, -77, -77, -77, -77, -77, -77, -77, -77, -77];

    let type1 = {type: nn.FLOAT32};
    let type34 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.25, zeroPoint: 0};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.00390625, zeroPoint: -128};
    let type4_length = product(type4.dimensions);
    let type6 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type34);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let axis = operandIndex++;
    model.addOperand(type6);
    let op2 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(param3, new Float32Array([1e-06]));
    model.setOperandValue(axis, new Int32Array([-1]));
    model.addOperation(nn.SOFTMAX, [op1, param3, axis], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Int8Array(type4_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op2_output[i], op2_expect[i]));
    }
  });
});
