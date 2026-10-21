using System;

namespace Neural_Network
{
    public static class DataHelper
    {
        public static double[][] MakeAllData(
            int numInput,
            int numHidden,
            int numOutput,
            int numRows,
            int seed)
        {
            if (numRows < 1)
                throw new ArgumentException("Number of rows must be positive.");

            Random random = new Random(seed);

            // A separate network generates the synthetic training targets.
            NeuralNetwork generator =
                new NeuralNetwork(numInput, numHidden, numOutput);

            int weightCount =
                (numInput * numHidden) + numHidden +
                (numHidden * numOutput) + numOutput;

            double[] weights = new double[weightCount];

            for (int i = 0; i < weights.Length; i++)
                weights[i] = 20.0 * random.NextDouble() - 10.0;

            generator.SetWeights(weights);

            double[][] data = new double[numRows][];

            for (int row = 0; row < numRows; row++)
            {
                double[] inputs = new double[numInput];

                // Generate input values between -10 and +10.
                for (int i = 0; i < numInput; i++)
                    inputs[i] = 20.0 * random.NextDouble() - 10.0;

                double[] outputs = generator.ComputeOutputs(inputs);

                // The largest output determines the target class.
                int targetClass = 0;

                for (int k = 1; k < numOutput; k++)
                {
                    if (outputs[k] > outputs[targetClass])
                        targetClass = k;
                }

                // Each row contains inputs followed by one-hot targets.
                data[row] = new double[numInput + numOutput];

                Array.Copy(inputs, data[row], numInput);

                data[row][numInput + targetClass] = 1.0;
            }

            return data;
        }

        public static void SplitTrainTest(
            double[][] allData,
            double trainFraction,
            int seed,
            out double[][] trainData,
            out double[][] testData)
        {
            if (allData == null || allData.Length < 2)
                throw new ArgumentException(
                    "At least two data rows are required.");

            if (double.IsNaN(trainFraction) ||
                trainFraction <= 0.0 || trainFraction >= 1.0)
            {
                throw new ArgumentException(
                    "Training fraction must be between 0 and 1, excluding both.");
            }

            int trainCount = (int)(allData.Length * trainFraction);
            int testCount = allData.Length - trainCount;

            if (trainCount == 0 || testCount == 0)
                throw new ArgumentException(
                    "The split must produce nonempty training and testing sets.");

            // Copy the outer array so shuffling preserves the original order.
            double[][] shuffled = (double[][])allData.Clone();

            Random random = new Random(seed);

            // Fisher-Yates shuffle.
            for (int i = 0; i < shuffled.Length; i++)
            {
                int randomIndex = random.Next(i, shuffled.Length);

                double[] temporary = shuffled[i];
                shuffled[i] = shuffled[randomIndex];
                shuffled[randomIndex] = temporary;
            }

            trainData = new double[trainCount][];
            testData = new double[testCount][];

            // Clone each row so the resulting datasets are independent.
            for (int i = 0; i < trainCount; i++)
                trainData[i] = (double[])shuffled[i].Clone();

            for (int i = 0; i < testCount; i++)
                testData[i] = (double[])shuffled[trainCount + i].Clone();
        }
    }
}