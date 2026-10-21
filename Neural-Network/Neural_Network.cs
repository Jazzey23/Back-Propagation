using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    public class NeuralNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights;
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights;
        private double[] oBiases;
        private double[] outputs;

        private Random rnd;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            if (numInput < 1 || numHidden < 1 || numOutput < 1)
                throw new ArgumentException("Layer sizes must be positive.");

            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];
            ihWeights = MakeMatrix(numInput, numHidden);
            hBiases = new double[numHidden];
            hOutputs = new double[numHidden];

            hoWeights = MakeMatrix(numHidden, numOutput);
            oBiases = new double[numOutput];
            outputs = new double[numOutput];

            rnd = new Random(0);
            InitializeWeights();
        }

        private static double[][] MakeMatrix(int rows, int columns)
        {
            double[][] result = new double[rows][];

            for (int i = 0; i < rows; i++)
                result[i] = new double[columns];

            return result;
        }

        private void InitializeWeights()
        {
            int count = (numInput * numHidden) + numHidden
                      + (numHidden * numOutput) + numOutput;

            double[] weights = new double[count];

            for (int i = 0; i < weights.Length; i++)
                weights[i] = 0.0009 * rnd.NextDouble() + 0.0001;

            SetWeights(weights);
        }

        public void SetWeights(double[] weights)
        {
            int count = (numInput * numHidden) + numHidden
                      + (numHidden * numOutput) + numOutput;

            if (weights == null || weights.Length != count)
                throw new ArgumentException("Incorrect number of weights.");

            int index = 0;

            for (int i = 0; i < numInput; i++)
                for (int j = 0; j < numHidden; j++)
                    ihWeights[i][j] = weights[index++];

            for (int j = 0; j < numHidden; j++)
                hBiases[j] = weights[index++];

            for (int j = 0; j < numHidden; j++)
                for (int k = 0; k < numOutput; k++)
                    hoWeights[j][k] = weights[index++];

            for (int k = 0; k < numOutput; k++)
                oBiases[k] = weights[index++];
        }

        public double[] GetWeights()
        {
            int count = (numInput * numHidden) + numHidden
                      + (numHidden * numOutput) + numOutput;

            double[] result = new double[count];
            int index = 0;

            for (int i = 0; i < numInput; i++)
                for (int j = 0; j < numHidden; j++)
                    result[index++] = ihWeights[i][j];

            for (int j = 0; j < numHidden; j++)
                result[index++] = hBiases[j];

            for (int j = 0; j < numHidden; j++)
                for (int k = 0; k < numOutput; k++)
                    result[index++] = hoWeights[j][k];

            for (int k = 0; k < numOutput; k++)
                result[index++] = oBiases[k];

            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            if (xValues == null || xValues.Length != numInput)
                throw new ArgumentException("Incorrect number of inputs.");

            Array.Copy(xValues, inputs, numInput);

            // Input layer to hidden layer.
            for (int j = 0; j < numHidden; j++)
            {
                double sum = hBiases[j];

                for (int i = 0; i < numInput; i++)
                    sum += inputs[i] * ihWeights[i][j];

                hOutputs[j] = Math.Tanh(sum);
            }

            // Hidden layer to output layer.
            double[] outputSums = new double[numOutput];

            for (int k = 0; k < numOutput; k++)
            {
                double sum = oBiases[k];

                for (int j = 0; j < numHidden; j++)
                    sum += hOutputs[j] * hoWeights[j][k];

                outputSums[k] = sum;
            }

            double[] probabilities = Softmax(outputSums);
            Array.Copy(probabilities, outputs, numOutput);

            return (double[])outputs.Clone();
        }

        private static double[] Softmax(double[] sums)
        {
            // Subtracting the maximum helps prevent exponential overflow.
            double maximum = sums[0];

            for (int i = 1; i < sums.Length; i++)
                if (sums[i] > maximum)
                    maximum = sums[i];

            double[] result = new double[sums.Length];
            double total = 0.0;

            for (int i = 0; i < sums.Length; i++)
            {
                result[i] = Math.Exp(sums[i] - maximum);
                total += result[i];
            }

            for (int i = 0; i < result.Length; i++)
                result[i] /= total;

            return result;
        }

        public double[] Train(
    double[][] trainData,
    int maxEpochs,
    double learnRate,
    double momentum,
    bool useCrossEntropy = false,
    double weightDecay = 0.0,
    IProgress<string> progress = null,
    string runName = "Network")
        {
            ValidateData(trainData);

            if (maxEpochs < 1)
                throw new ArgumentException("Epochs must be positive.");

            if (double.IsNaN(learnRate) ||
                double.IsInfinity(learnRate) ||
                learnRate <= 0.0)
            {
                throw new ArgumentException(
                    "Learning rate must be positive and finite.");
            }

            if (double.IsNaN(momentum) ||
                double.IsInfinity(momentum) ||
                momentum < 0.0 || momentum >= 1.0)
            {
                throw new ArgumentException(
                    "Momentum must be between 0 and 1, excluding 1.");
            }

            if (double.IsNaN(weightDecay) ||
                double.IsInfinity(weightDecay) ||
                weightDecay < 0.0)
            {
                throw new ArgumentException(
                    "Weight decay must be nonnegative and finite.");
            }

            double[] outputSignals = new double[numOutput];
            double[] hiddenSignals = new double[numHidden];

            double[][] ihPreviousDelta = MakeMatrix(numInput, numHidden);
            double[][] hoPreviousDelta = MakeMatrix(numHidden, numOutput);
            double[] hPreviousDelta = new double[numHidden];
            double[] oPreviousDelta = new double[numOutput];

            double[] xValues = new double[numInput];
            double[] targets = new double[numOutput];

            int[] sequence = new int[trainData.Length];

            for (int i = 0; i < sequence.Length; i++)
                sequence[i] = i;

            int reportInterval = Math.Max(1, maxEpochs / 10);

            double initialLoss = CalculateLoss(trainData, useCrossEntropy);
            double initialPenalty = GetWeightPenalty(weightDecay);

            if (progress != null)
            {
                progress.Report(
                    runName + " | Initial loss: "
                    + initialLoss.ToString("F6")
                    + " | Objective: "
                    + (initialLoss + initialPenalty).ToString("F6"));
            }

            for (int epoch = 1; epoch <= maxEpochs; epoch++)
            {
                Shuffle(sequence);

                for (int row = 0; row < sequence.Length; row++)
                {
                    int index = sequence[row];

                    Array.Copy(trainData[index], xValues, numInput);
                    Array.Copy(
                        trainData[index], numInput,
                        targets, 0, numOutput);

                    ComputeOutputs(xValues);

                    // 1. Calculate output signals.
                    for (int k = 0; k < numOutput; k++)
                    {
                        double error = targets[k] - outputs[k];

                        if (useCrossEntropy)
                        {
                            // Softmax + cross-entropy.
                            // Updates are added, so use target - prediction.
                            outputSignals[k] = error;
                        }
                        else
                        {
                            // Preserve the supplied baseline calculation.
                            // This is not the full softmax/squared-error gradient.
                            outputSignals[k] =
                                error * outputs[k] * (1.0 - outputs[k]);
                        }
                    }

                    // 2. Calculate hidden signals BEFORE changing any weights.
                    for (int j = 0; j < numHidden; j++)
                    {
                        double sum = 0.0;

                        for (int k = 0; k < numOutput; k++)
                            sum += outputSignals[k] * hoWeights[j][k];

                        double derivative =
                            (1.0 + hOutputs[j]) * (1.0 - hOutputs[j]);

                        hiddenSignals[j] = derivative * sum;
                    }

                    // 3. Update input-to-hidden weights.
                    for (int i = 0; i < numInput; i++)
                    {
                        for (int j = 0; j < numHidden; j++)
                        {
                            double direction =
                                hiddenSignals[j] * inputs[i]
                                - weightDecay * ihWeights[i][j];

                            double delta = learnRate * direction;

                            ihWeights[i][j] +=
                                delta + momentum * ihPreviousDelta[i][j];

                            ihPreviousDelta[i][j] = delta;
                        }
                    }

                    // 4. Update hidden biases. No weight decay on biases.
                    for (int j = 0; j < numHidden; j++)
                    {
                        double delta = learnRate * hiddenSignals[j];

                        hBiases[j] +=
                            delta + momentum * hPreviousDelta[j];

                        hPreviousDelta[j] = delta;
                    }

                    // 5. Update hidden-to-output weights.
                    for (int j = 0; j < numHidden; j++)
                    {
                        for (int k = 0; k < numOutput; k++)
                        {
                            double direction =
                                outputSignals[k] * hOutputs[j]
                                - weightDecay * hoWeights[j][k];

                            double delta = learnRate * direction;

                            hoWeights[j][k] +=
                                delta + momentum * hoPreviousDelta[j][k];

                            hoPreviousDelta[j][k] = delta;
                        }
                    }

                    // 6. Update output biases. No weight decay on biases.
                    for (int k = 0; k < numOutput; k++)
                    {
                        double delta = learnRate * outputSignals[k];

                        oBiases[k] +=
                            delta + momentum * oPreviousDelta[k];

                        oPreviousDelta[k] = delta;
                    }
                }

                if (epoch % reportInterval == 0 || epoch == maxEpochs)
                {
                    double dataLoss = CalculateLoss(trainData, useCrossEntropy);
                    double penalty = GetWeightPenalty(weightDecay);
                    double objective = dataLoss + penalty;

                    if (double.IsNaN(objective) || double.IsInfinity(objective))
                    {
                        throw new InvalidOperationException(
                            "Training became unstable. Try a lower learning rate.");
                    }

                    if (progress != null)
                    {
                        progress.Report(
                            runName
                            + " | Epoch " + epoch
                            + " | Loss: " + dataLoss.ToString("F6")
                            + " | L2 penalty: " + penalty.ToString("F6")
                            + " | Objective: " + objective.ToString("F6"));
                    }
                }
            }

            return GetWeights();
        }

        public double CalculateLoss(double[][] data, bool useCrossEntropy)
        {
            ValidateData(data);

            double totalLoss = 0.0;
            double[] xValues = new double[numInput];

            for (int row = 0; row < data.Length; row++)
            {
                Array.Copy(data[row], xValues, numInput);

                double[] predicted = ComputeOutputs(xValues);

                if (useCrossEntropy)
                {
                    totalLoss += CrossEntropyForCurrentOutput(data[row]);
                }
                else
                {
                    for (int k = 0; k < numOutput; k++)
                    {
                        double target = data[row][numInput + k];
                        double difference = target - predicted[k];

                        totalLoss += difference * difference;
                    }
                }
            }

            // Matches the supplied baseline's squared-error normalization:
            // sum over output nodes, then average over examples.
            return totalLoss / data.Length;
        }

        public double GetWeightPenalty(double weightDecay)
        {
            double sumSquares = 0.0;

            for (int i = 0; i < numInput; i++)
                for (int j = 0; j < numHidden; j++)
                    sumSquares += ihWeights[i][j] * ihWeights[i][j];

            for (int j = 0; j < numHidden; j++)
                for (int k = 0; k < numOutput; k++)
                    sumSquares += hoWeights[j][k] * hoWeights[j][k];

            // Biases are intentionally excluded.
            return 0.5 * weightDecay * sumSquares;
        }

        private double CrossEntropyForCurrentOutput(double[] dataRow)
        {
            // Reconstruct output scores using the hidden activations
            // from the immediately preceding ComputeOutputs call.
            double[] scores = new double[numOutput];

            for (int k = 0; k < numOutput; k++)
            {
                scores[k] = oBiases[k];

                for (int j = 0; j < numHidden; j++)
                    scores[k] += hOutputs[j] * hoWeights[j][k];
            }

            double maximum = scores[0];

            for (int k = 1; k < numOutput; k++)
                if (scores[k] > maximum)
                    maximum = scores[k];

            double exponentialSum = 0.0;

            for (int k = 0; k < numOutput; k++)
                exponentialSum += Math.Exp(scores[k] - maximum);

            double logSum = Math.Log(exponentialSum);
            double loss = 0.0;

            for (int k = 0; k < numOutput; k++)
            {
                double target = dataRow[numInput + k];

                // log softmax(score[k]) calculated stably.
                double logProbability = (scores[k] - maximum) - logSum;

                loss -= target * logProbability;
            }

            return loss;
        }

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; i++)
            {
                int randomIndex = rnd.Next(i, sequence.Length);

                int temporary = sequence[randomIndex];
                sequence[randomIndex] = sequence[i];
                sequence[i] = temporary;
            }
        }

        private double Error(double[][] data)
        {
            // Sum of squared output errors, averaged over examples.
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput];

            for (int row = 0; row < data.Length; row++)
            {
                Array.Copy(data[row], xValues, numInput);
                double[] predicted = ComputeOutputs(xValues);

                for (int k = 0; k < numOutput; k++)
                {
                    double target = data[row][numInput + k];
                    double difference = target - predicted[k];

                    sumSquaredError += difference * difference;
                }
            }

            return sumSquaredError / data.Length;
        }

        public double Accuracy(double[][] data)
        {
            ValidateData(data);

            int correct = 0;
            double[] xValues = new double[numInput];
            double[] targets = new double[numOutput];

            for (int row = 0; row < data.Length; row++)
            {
                Array.Copy(data[row], xValues, numInput);
                Array.Copy(data[row], numInput, targets, 0, numOutput);

                double[] predicted = ComputeOutputs(xValues);

                if (MaxIndex(predicted) == MaxIndex(targets))
                    correct++;
            }

            // Returns a fraction, for example 0.95 for 95%.
            return (double)correct / data.Length;
        }

        private static int MaxIndex(double[] values)
        {
            int index = 0;

            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[index])
                    index = i;

            return index;
        }

        private void ValidateData(double[][] data)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data must contain at least one row.");

            int expectedColumns = numInput + numOutput;

            for (int row = 0; row < data.Length; row++)
            {
                if (data[row] == null ||
                    data[row].Length != expectedColumns)
                {
                    throw new ArgumentException(
                        "Each row must contain " + numInput +
                        " inputs followed by " + numOutput + " targets.");
                }
            }
        }
    }
}
