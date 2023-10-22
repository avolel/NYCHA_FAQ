using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Models;

namespace BuilderModel
{
	public class NYCHAFAQModelBuilder
	{
		private MLContext context;
		private IEnumerable<NYCHAFAQModel> trainingData;
		private IEnumerable<NYCHAFAQModel> testingData;
		private TransformerChain<KeyToValueMappingTransformer> model;
		private string fileName = "NYCHAFAQModel.zip";

		public NYCHAFAQModelBuilder(MLContext _context)
		{
			this.context = _context;
		}

		public void TrainModel()
		{
			context.Log += new EventHandler<LoggingEventArgs>(LogMLEvents);

			//Load data from csv file
			var data = context.Data.LoadFromTextFile<NYCHAFAQModel>("NYCHA_FAQ.csv", hasHeader: true, separatorChar: ',', allowQuoting: true, allowSparse: true, trimWhitespace: true);


			//create data sets for trainiing and testing
			trainingData = context.Data.CreateEnumerable<NYCHAFAQModel>(data, reuseRowObject: false);
			testingData = trainingData.Skip(Math.Max(0,trainingData.Count() - 20));
			
			//Create our pipeline and set our training model
			var pipeline = context.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Answer") //converts string to key value for training
				.Append(context.Transforms.Text.FeaturizeText("Features", "Question")) //creates features from our text string
				.Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))//set up our model
				.Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedAnswer", inputColumnName: "PredictedLabel")); //convert our key back to a label

			//trains the model
			model = pipeline.Fit(context.Data.LoadFromEnumerable(trainingData));
		}

		public void TestModel()
		{
			//transform data to a view that can be evaluated
			IDataView testDataPredictions = model.Transform(context.Data.LoadFromEnumerable(testingData));
			//evaluate test data against trained model for accuracy
			var metrics = context.MulticlassClassification.Evaluate(testDataPredictions);
			double accuracy = metrics.MacroAccuracy;

			Console.WriteLine("Accuracy {0}", accuracy.ToString());
		}

		public void SaveModel()
		{
			IDataView dataView = context.Data.LoadFromEnumerable<NYCHAFAQModel>(trainingData);
			context.Model.Save(model, dataView.Schema, fileName);
		}

		private static void LogMLEvents(object sender, LoggingEventArgs e) =>
			Console.WriteLine(e.RawMessage);
	}
}