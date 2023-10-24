using Microsoft.ML.Data;

namespace BuilderModel
{
	public class PredictionModel
	{
		[ColumnName("PredictedAnswer")]
		public string PredictedAnswer { get; set; }
		[ColumnName("Score")]
		public float[] Score { get; set; }
	}
}