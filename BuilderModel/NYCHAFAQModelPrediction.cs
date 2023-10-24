using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace BuilderModel
{
	public class NYCHAFAQModelPrediction
	{
		private MLContext context;
		private TransformerChain<KeyToValueMappingTransformer> model;
		private string fileName = "NYCHAFAQModel.zip";

		public NYCHAFAQModelPrediction()
		{
			this.context = new MLContext();
		}

		public PredictionModel Predict(NYCHAFAQModel Question)
		{
			ITransformer trainedModel = LoadModel();

			//Creates prediction function from loaded model, you can load in memory model as wwell
			var predictFunction = context.Model.CreatePredictionEngine<NYCHAFAQModel, PredictionModel>(trainedModel);

			//pass model to function to get prediction outputs
			PredictionModel prediction = predictFunction.Predict(Question);

			//get score, score is an array and the max score will align to key.
			float score = prediction.Score.Max();

			return prediction;
		}

		private ITransformer LoadModel()
		{
			DataViewSchema modelSchema;
			//gets a file from a stream, and loads it
			using (Stream s = File.Open(fileName, FileMode.Open))
			{
				return context.Model.Load(s, out modelSchema);
			}
		}
	}
}