using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Models;
using NYCHA_FAQPredictiveModel;
using BuilderModel;

namespace NYCHA_FAQ.Controllers
{
	[ApiController]
	[Route("[controller]")]
	public class NychaFAQController : ControllerBase
	{
		private readonly ILogger<NychaFAQController> _logger;

		public NychaFAQController(ILogger<NychaFAQController> logger)
		{
			_logger = logger;
		}

		[HttpGet(Name = "GetPrediction")]
		public PredictionModel Get(string message)
		{
			MLContext context = new MLContext();
			NYCHAFAQModelPrediction predict = new NYCHAFAQModelPrediction(context);
			NYCHAFAQModel question = new NYCHAFAQModel()
			{
				Question = message,
				Answer = ""
			};
			return predict.Predict(question);
		}
	}
}
