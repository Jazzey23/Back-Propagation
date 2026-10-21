using System;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Network
{
    public partial class Form1 : Form
    {
        private NeuralNetwork trainedNetwork;

        public Form1()
        {
            InitializeComponent();

            Text = "Back-Propagation Comparison";
            btnTrain.Text = "Run Comparison";
            btnTrain.Click += btnTrain_Click;
        }

        private async void btnTrain_Click(object sender, EventArgs e)
        {
            btnTrain.Enabled = false;
            trainedNetwork = null;
            txtResults.Clear();

            const int numInput = 4;
            const int numHidden = 5;
            const int numOutput = 3;
            const int numRows = 1000;
            const int seed = 1;

            const int maxEpochs = 1000;
            const double learningRate = 0.05;
            const double momentum = 0.01;
            const double weightDecay = 0.001;

            txtResults.AppendText(
                "BACK-PROPAGATION COMPARISON" + Environment.NewLine
                + "Network: 4 inputs, 5 hidden neurons, 3 outputs"
                + Environment.NewLine
                + "Data: 800 training / 200 testing examples"
                + Environment.NewLine
                + "Epochs: 1000 | Learning rate: 0.05 | Momentum: 0.01"
                + Environment.NewLine
                + "Modified model weight decay: 0.001"
                + Environment.NewLine
                + Environment.NewLine
                + "Both runs use the same data and starting weights."
                + Environment.NewLine
                + "Baseline loss: squared error per example."
                + Environment.NewLine
                + "Modified loss: cross-entropy per example."
                + Environment.NewLine
                + "Modified objective: cross-entropy + L2 penalty."
                + Environment.NewLine
                + Environment.NewLine);

            // Created on the UI thread so progress safely updates controls.
            IProgress<string> progress = new Progress<string>(message =>
            {
                if (!IsDisposed && !Disposing)
                    txtResults.AppendText(message + Environment.NewLine);
            });

            try
            {
                var result = await Task.Run(() =>
                {
                    double[][] allData = DataHelper.MakeAllData(
                        numInput, numHidden, numOutput, numRows, seed);

                    double[][] trainData;
                    double[][] testData;

                    DataHelper.SplitTrainTest(
                        allData, 0.80, seed,
                        out trainData, out testData);

                    // Both constructors use the same random seed.
                    // Both models therefore also shuffle training rows
                    // in the same order during their separate runs.
                    NeuralNetwork baseline = new NeuralNetwork(
                        numInput, numHidden, numOutput);

                    NeuralNetwork modified = new NeuralNetwork(
                        numInput, numHidden, numOutput);

                    // Explicitly give both models identical initial weights.
                    double[] startingWeights = baseline.GetWeights();
                    modified.SetWeights(startingWeights);

                    progress.Report("Starting supplied baseline...");

                    baseline.Train(
                        trainData,
                        maxEpochs,
                        learningRate,
                        momentum,
                        useCrossEntropy: false,
                        weightDecay: 0.0,
                        progress: progress,
                        runName: "Baseline");

                    progress.Report("");
                    progress.Report(
                        "Starting cross-entropy with weight decay...");

                    modified.Train(
                        trainData,
                        maxEpochs,
                        learningRate,
                        momentum,
                        useCrossEntropy: true,
                        weightDecay: weightDecay,
                        progress: progress,
                        runName: "Modified");

                    StringBuilder report = new StringBuilder();

                    report.AppendLine();
                    report.AppendLine("FINAL COMPARISON");
                    report.AppendLine(new string('=', 68));

                    report.AppendLine(string.Format(
                        "{0,-30}{1,17}{2,17}",
                        "Metric", "Baseline", "Modified"));

                    report.AppendLine(new string('-', 68));

                    double baselineTrainAccuracy =
                        baseline.Accuracy(trainData);

                    double modifiedTrainAccuracy =
                        modified.Accuracy(trainData);

                    double baselineTestAccuracy =
                        baseline.Accuracy(testData);

                    double modifiedTestAccuracy =
                        modified.Accuracy(testData);

                    report.AppendLine(string.Format(
                        "{0,-30}{1,17:P2}{2,17:P2}",
                        "Training accuracy",
                        baselineTrainAccuracy,
                        modifiedTrainAccuracy));

                    report.AppendLine(string.Format(
                        "{0,-30}{1,17:P2}{2,17:P2}",
                        "Testing accuracy",
                        baselineTestAccuracy,
                        modifiedTestAccuracy));

                    // Evaluate BOTH models with BOTH losses.
                    // This allows comparisons using the same metric.
                    report.AppendLine(string.Format(
                        "{0,-30}{1,17:F6}{2,17:F6}",
                        "Training squared error",
                        baseline.CalculateLoss(trainData, false),
                        modified.CalculateLoss(trainData, false)));

                    report.AppendLine(string.Format(
                        "{0,-30}{1,17:F6}{2,17:F6}",
                        "Testing squared error",
                        baseline.CalculateLoss(testData, false),
                        modified.CalculateLoss(testData, false)));

                    double baselineTrainCE =
                        baseline.CalculateLoss(trainData, true);

                    double modifiedTrainCE =
                        modified.CalculateLoss(trainData, true);

                    report.AppendLine(string.Format(
                        "{0,-30}{1,17:F6}{2,17:F6}",
                        "Training cross-entropy",
                        baselineTrainCE,
                        modifiedTrainCE));

                    report.AppendLine(string.Format(
                        "{0,-30}{1,17:F6}{2,17:F6}",
                        "Testing cross-entropy",
                        baseline.CalculateLoss(testData, true),
                        modified.CalculateLoss(testData, true)));

                    double penalty = modified.GetWeightPenalty(weightDecay);

                    report.AppendLine(string.Format(
                        "{0,-30}{1,17:F6}{2,17:F6}",
                        "Applied L2 penalty",
                        0.0,
                        penalty));

                    report.AppendLine();
                    report.AppendLine(
                        "Modified final training objective: "
                        + (modifiedTrainCE + penalty).ToString("F6"));

                    double accuracyChange =
                        (modifiedTestAccuracy - baselineTestAccuracy) * 100.0;

                    report.AppendLine(
                        "Test accuracy change: "
                        + accuracyChange.ToString("+0.00;-0.00;0.00")
                        + " percentage points.");

                    report.AppendLine();
                    report.AppendLine(
                        "Compare values within the same metric row.");
                    report.AppendLine(
                        "A lower loss is better; a higher accuracy is better.");
                    report.AppendLine(
                        "The modified model is not guaranteed to outperform");
                    report.AppendLine(
                        "the baseline on this particular data split.");

                    report.AppendLine();
                    report.AppendLine(
                        "MODIFIED MODEL: FINAL WEIGHTS AND BIASES");

                    double[] finalWeights = modified.GetWeights();

                    for (int i = 0; i < finalWeights.Length; i++)
                    {
                        report.AppendLine(
                            "[" + i.ToString("D2") + "] "
                            + finalWeights[i].ToString("F6"));
                    }

                    return new
                    {
                        Network = modified,
                        Report = report.ToString()
                    };
                });

                if (IsDisposed || Disposing)
                    return;

                trainedNetwork = result.Network;
                txtResults.AppendText(result.Report);
            }
            catch (Exception ex)
            {
                if (!IsDisposed && !Disposing)
                {
                    txtResults.AppendText(
                        Environment.NewLine
                        + "Comparison failed: "
                        + ex.Message
                        + Environment.NewLine);
                }
            }
            finally
            {
                if (!IsDisposed && !Disposing)
                    btnTrain.Enabled = true;
            }
        }
    }
}