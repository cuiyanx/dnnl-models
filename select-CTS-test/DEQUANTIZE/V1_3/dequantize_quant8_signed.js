// Generated file (from: dequantize_quant8_signed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Dequantize quant8 signed example-1', async function() {
    // For 'Dequantize quant8 signed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-128, -96, 0, 127];
    let op2_expect = [0.0, 32.0, 128.0, 255.0];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 1], scale: 1.0, zeroPoint: -128};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);

    model.addOperation(nn.DEQUANTIZE, [op1], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Dequantize quant8 signed example-2', async function() {
    // For 'Dequantize quant8 signed' example: examples_1d_quant8_asymm
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input0_value = [-128, -127, -126, -125, -124, 123, 124, 125, 126, 127];
    let output0_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type2 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [10], scale: 0.5, zeroPoint: -1};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [10]};
    let type3_length = product(type3.dimensions);

    let input0 = operandIndex++;
    model.addOperand(type2);
    let output0 = operandIndex++;
    model.addOperand(type3);

    model.addOperation(nn.DEQUANTIZE, [input0], [output0]);

    model.identifyInputsAndOutputs([input0], [output0]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input0_input = new Int8Array(input0_value);
    execution.setInput(0, input0_input);
    let output0_output = new Float32Array(type3_length);
    execution.setOutput(0, output0_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output0_output[i], output0_expect[i]));
    }
  });

  it('check result for Dequantize quant8 signed example-3', async function() {
    // For 'Dequantize quant8 signed' example: examples_1d_quant8_asymm_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input0_value = [-128, -127, -126, -125, -124, 123, 124, 125, 126, 127];
    let output0_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type2 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [10], scale: 0.5, zeroPoint: -1};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [10]};
    let type3_length = product(type3.dimensions);

    let input0 = operandIndex++;
    model.addOperand(type2);
    let output0 = operandIndex++;
    model.addOperand(type3);

    model.addOperation(nn.DEQUANTIZE, [input0], [output0]);

    model.identifyInputsAndOutputs([input0], [output0]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input0_input = new Int8Array(input0_value);
    execution.setInput(0, input0_input);
    let output0_output = new Float32Array(type3_length);
    execution.setOutput(0, output0_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output0_output[i], output0_expect[i]));
    }
  });

  it('check result for Dequantize quant8 signed example-4', async function() {
    // For 'Dequantize quant8 signed' example: examples_1d_quant8_asymm_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input0_value = [-128, -127, -126, -125, -124, 123, 124, 125, 126, 127];
    let output0_expect = [-63.5, -63.0, -62.5, -62.0, -61.5, 62.0, 62.5, 63.0, 63.5, 64.0];

    let type19 = {type: nn.TENSOR_FLOAT32, dimensions: [10]};
    let type19_length = product(type19.dimensions);
    let type2 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [10], scale: 0.5, zeroPoint: -1};
    let type2_length = product(type2.dimensions);

    let input0 = operandIndex++;
    model.addOperand(type2);
    let output0 = operandIndex++;
    model.addOperand(type19);

    model.addOperation(nn.DEQUANTIZE, [input0], [output0]);

    model.identifyInputsAndOutputs([input0], [output0]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input0_input = new Int8Array(input0_value);
    execution.setInput(0, input0_input);
    let output0_output = new Float32Array(type19_length);
    execution.setOutput(0, output0_output);

    await execution.startCompute();

    for (let i = 0; i < type19_length; ++i) {
      assert.isTrue(almostEqualCTS(output0_output[i], output0_expect[i]));
    }
  });

  it('check result for Dequantize quant8 signed example-5', async function() {
    // For 'Dequantize quant8 signed' example: examples_2d_quant8_asymm
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input01_value = [-128, -127, -126, -125, -124, 123, 124, 125, 126, 127];
    let output01_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.5, zeroPoint: -1};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 5]};
    let type5_length = product(type5.dimensions);

    let input01 = operandIndex++;
    model.addOperand(type4);
    let output01 = operandIndex++;
    model.addOperand(type5);

    model.addOperation(nn.DEQUANTIZE, [input01], [output01]);

    model.identifyInputsAndOutputs([input01], [output01]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input01_input = new Int8Array(input01_value);
    execution.setInput(0, input01_input);
    let output01_output = new Float32Array(type5_length);
    execution.setOutput(0, output01_output);

    await execution.startCompute();

    for (let i = 0; i < type5_length; ++i) {
      assert.isTrue(almostEqualCTS(output01_output[i], output01_expect[i]));
    }
  });

  it('check result for Dequantize quant8 signed example-6', async function() {
    // For 'Dequantize quant8 signed' example: examples_2d_quant8_asymm_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input01_value = [-128, -127, -126, -125, -124, 123, 124, 125, 126, 127];
    let output01_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.5, zeroPoint: -1};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 5]};
    let type5_length = product(type5.dimensions);

    let input01 = operandIndex++;
    model.addOperand(type4);
    let output01 = operandIndex++;
    model.addOperand(type5);

    model.addOperation(nn.DEQUANTIZE, [input01], [output01]);

    model.identifyInputsAndOutputs([input01], [output01]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input01_input = new Int8Array(input01_value);
    execution.setInput(0, input01_input);
    let output01_output = new Float32Array(type5_length);
    execution.setOutput(0, output01_output);

    await execution.startCompute();

    for (let i = 0; i < type5_length; ++i) {
      assert.isTrue(almostEqualCTS(output01_output[i], output01_expect[i]));
    }
  });

  it('check result for Dequantize quant8 signed example-7', async function() {
    // For 'Dequantize quant8 signed' example: examples_2d_quant8_asymm_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input01_value = [-128, -127, -126, -125, -124, 123, 124, 125, 126, 127];
    let output01_expect = [-63.5, -63.0, -62.5, -62.0, -61.5, 62.0, 62.5, 63.0, 63.5, 64.0];

    let type20 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 5]};
    let type20_length = product(type20.dimensions);
    let type4 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [2, 5], scale: 0.5, zeroPoint: -1};
    let type4_length = product(type4.dimensions);

    let input01 = operandIndex++;
    model.addOperand(type4);
    let output01 = operandIndex++;
    model.addOperand(type20);

    model.addOperation(nn.DEQUANTIZE, [input01], [output01]);

    model.identifyInputsAndOutputs([input01], [output01]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input01_input = new Int8Array(input01_value);
    execution.setInput(0, input01_input);
    let output01_output = new Float32Array(type20_length);
    execution.setOutput(0, output01_output);

    await execution.startCompute();

    for (let i = 0; i < type20_length; ++i) {
      assert.isTrue(almostEqualCTS(output01_output[i], output01_expect[i]));
    }
  });

  it('check result for Dequantize quant8 signed example-8', async function() {
    // For 'Dequantize quant8 signed' example: examples_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [-128, -96, 0, 127];
    let op21_expect = [0.0, 32.0, 128.0, 255.0];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 1], scale: 1.0, zeroPoint: -128};
    let type0_length = product(type0.dimensions);
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type0);
    let op21 = operandIndex++;
    model.addOperand(type6);

    model.addOperation(nn.DEQUANTIZE, [op11], [op21]);

    model.identifyInputsAndOutputs([op11], [op21]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Int8Array(op11_value);
    execution.setInput(0, op11_input);
    let op21_output = new Float32Array(type6_length);
    execution.setOutput(0, op21_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTS(op21_output[i], op21_expect[i]));
    }
  });
});
