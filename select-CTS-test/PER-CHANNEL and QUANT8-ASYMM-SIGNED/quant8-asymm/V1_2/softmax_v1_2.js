// Generated file (from: softmax_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Softmax v1_2 example-1', async function() {
    // For 'Softmax v1_2' example: examples_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [196, 192, 188, 184, 132, 124, 120, 116, 112, 60, 196, 192, 188, 184, 132, 124, 120, 116, 112, 60, 196, 192, 188, 184, 132, 124, 120, 116, 112, 60, 196, 192, 188, 184, 132, 124, 120, 116, 112, 60];
    let op2_expect = [165, 61, 22, 8, 0, 165, 61, 22, 8, 0, 165, 61, 22, 8, 0, 165, 61, 22, 8, 0, 165, 61, 22, 8, 0, 165, 61, 22, 8, 0, 165, 61, 22, 8, 0, 165, 61, 22, 8, 0];

    let type19 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 128};
    let type19_length = product(type19.dimensions);
    let type2 = {type: nn.FLOAT32};
    let type20 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: 0};
    let type20_length = product(type20.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type19);
    let param = operandIndex++;
    model.addOperand(type2);
    let op2 = operandIndex++;
    model.addOperand(type20);

    model.setOperandValue(param, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [op1, param], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Uint8Array(type20_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type20_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });

  it('check result for Softmax v1_2 example-2', async function() {
    // For 'Softmax v1_2' example: examples_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [132, 136, 140, 144, 148, 124, 120, 116, 112, 108, 132, 136, 140, 144, 148, 124, 120, 116, 112, 108, 132, 136, 140, 144, 148, 124, 120, 116, 112, 108, 132, 136, 140, 144, 148, 124, 120, 116, 112, 108];
    let op2_expect = [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51];

    let type19 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 2, 2, 5], scale: 0.25, zeroPoint: 128};
    let type19_length = product(type19.dimensions);
    let type2 = {type: nn.FLOAT32};
    let type20 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 2, 2, 5], scale: 0.00390625, zeroPoint: 0};
    let type20_length = product(type20.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type19);
    let param1 = operandIndex++;
    model.addOperand(type2);
    let op2 = operandIndex++;
    model.addOperand(type20);

    model.setOperandValue(param1, new Float32Array([1e-06]));
    model.addOperation(nn.SOFTMAX, [op1, param1], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Uint8Array(type20_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type20_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });
});
