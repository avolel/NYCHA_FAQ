﻿using Microsoft.ML.Data;

namespace BuilderModel
{
	public class NYCHAFAQModel
	{
		[ColumnName("Question"), LoadColumn(0)]
		public string Question { get; set; }
		[ColumnName("Answer"), LoadColumn(1)]
		public string Answer { get; set; }
	}
}