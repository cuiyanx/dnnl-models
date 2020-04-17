// Generated file (from: dequantize.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Dequantize example', async function() {
    // For 'Dequantize' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [0, 32, 128, 255];
    let op2_expect = [0.0, 32.0, 128.0, 255.0];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 1.0, zeroPoint: 0};
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

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });
});
