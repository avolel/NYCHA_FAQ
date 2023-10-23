using BuilderModel;

NYCHAFAQModelBuilder nychaFAQ = new NYCHAFAQModelBuilder();

nychaFAQ.TrainModel();
nychaFAQ.TestModel();
nychaFAQ.SaveModel();