using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Models;

namespace BuilderModel
{
	public class NYCHAFAQModelBuilder
	{
		private MLContext context;
		private IDataView trainingData;
		private IDataView testingData;
		private TransformerChain<KeyToValueMappingTransformer> model;
		private string fileName = "NYCHAFAQModel.zip";

		public NYCHAFAQModelBuilder()
		{
			this.context = new MLContext();
		}

		public void TrainModel()
		{
			context.Log += new EventHandler<LoggingEventArgs>(LogMLEvents);

			//Load data from csv file
			IDataView data = context.Data.LoadFromTextFile<NYCHAFAQModel>("NYCHA_FAQ.csv", hasHeader: false, separatorChar: ',');


			//create data sets for trainiing and testing
			DataOperationsCatalog.TrainTestData dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1234, samplingKeyColumnName: "Answer");
			trainingData = dataSplit.TrainSet;
			testingData = dataSplit.TestSet;
			
			//Create our pipeline and set our training model
			var pipeline = context.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Answer") //converts string to key value for training
				.Append(context.Transforms.Text.FeaturizeText("Features", "Question")) //creates features from our text string
				.Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))//set up our model
				.Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedAnswer", inputColumnName: "PredictedLabel")); //convert our key back to a label

			//trains the model
			model = pipeline.Fit(trainingData);
		}

		public void TestModel()
		{
			//transform data to a view that can be evaluated
			IDataView testDataPredictions = model.Transform(testingData);
			//evaluate test data against trained model for accuracy
			MulticlassClassificationMetrics metrics = context.MulticlassClassification.Evaluate(testDataPredictions);

			Console.WriteLine($"Macro Accuracy {metrics.MacroAccuracy}");
			Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");
			Console.WriteLine($"Log Loss: {metrics.LogLoss}");
			Console.WriteLine();
			Console.WriteLine();
			Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
		}

		public void SaveModel() =>
			context.Model.Save(model, trainingData.Schema, fileName);

		private static void LogMLEvents(object sender, LoggingEventArgs e) =>
			Console.WriteLine(e.RawMessage);
	}
}