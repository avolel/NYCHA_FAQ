using Microsoft.ML;
using BuilderModel;

MLContext context = new MLContext();
NYCHAFAQModelBuilder nychaFAQ = new NYCHAFAQModelBuilder(context);

nychaFAQ.TrainModel();
nychaFAQ.TestModel();
nychaFAQ.SaveModel();