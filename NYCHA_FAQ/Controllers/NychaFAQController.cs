using Microsoft.AspNetCore.Mvc;
using Models;
using NYCHA_FAQPredictiveModel;

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

		[HttpPost(Name = "GetPrediction")]
		public PredictionModel PostGetPrediction([FromBody]string message)
		{
			NYCHAFAQModelPrediction predict = new NYCHAFAQModelPrediction();
			NYCHAFAQModel question = new NYCHAFAQModel()
			{
				Question = message,
				Answer = ""
			};
			return predict.Predict(question);
		}
	}
}
