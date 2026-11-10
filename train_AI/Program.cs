using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace HebrewDateTrainer;

/// <summary>One row of the CSV the data generator wrote.</summary>
public class DateRow
{
    [LoadColumn(0)] public string Text { get; set; } = "";
    [LoadColumn(1)] public string Day { get; set; } = "";
    [LoadColumn(2)] public string Month { get; set; } = "";
    [LoadColumn(3)] public string Year { get; set; } = "";
    // The year's three letter-digits: תשפ״ז = 7, 8, 7. All 0 when no year was typed.
    [LoadColumn(4)] public string YearHundreds { get; set; } = "";
    [LoadColumn(5)] public string YearTens { get; set; } = "";
    [LoadColumn(6)] public string YearUnits { get; set; } = "";
}

/// <summary>What a model answers: its guess and how sure it is of each option.</summary>
public class Guess
{
    public string PredictedValue { get; set; } = "";
    public float[] Score { get; set; } = [];
}

class Program
{
    static readonly string DataPath = Path.Combine("..", "create_examples", "data", "hebrew_dates.csv");
    const string ModelsFolder = "models";

    // The server ignores answers below this confidence. Keep both values the same.
    const float MinConfidence = 0.8f;

    static readonly string[] RegularYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "Adar",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];
    static readonly string[] LeapYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "AdarI", "AdarII",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];

    static void Main()
    {
        Console.InputEncoding = Encoding.UTF8;
        Console.OutputEncoding = Encoding.UTF8;

        if (!File.Exists(DataPath))
        {
            Console.WriteLine($"Training data not found at: {Path.GetFullPath(DataPath)}");
            Console.WriteLine("Run the data generator first (dotnet run in its folder).");
            return;
        }

        var ml = new MLContext(seed: 1);
        IDataView data = ml.Data.LoadFromTextFile<DateRow>(
            DataPath, separatorChar: ',', hasHeader: true, allowQuoting: true);

        // Small models, one per part of the date. The year gets three models, one per
        // letter-digit, because one model with ~290 years was slow to train.
        var dayModel = LoadOrTrain(ml, data, "Day");
        var monthModel = LoadOrTrain(ml, data, "Month");
        var hundredsModel = LoadOrTrain(ml, data, "YearHundreds");
        var tensModel = LoadOrTrain(ml, data, "YearTens");
        var unitsModel = LoadOrTrain(ml, data, "YearUnits");
        TestFullYear(ml, data, hundredsModel, tensModel, unitsModel);

        TryItYourself(ml, dayModel, monthModel, hundredsModel, tensModel, unitsModel);
    }

    /// <summary>
    /// Builds the year from the three digit answers: 7, 8, 7 → 5787. "0" means no year typed.
    /// The year is only as sure as its least sure digit.
    /// </summary>
    static (string Year, float Sure) CombineYear(Guess hundreds, Guess tens, Guess units)
    {
        // Hundreds 0 means no year was typed; the other two digits don't matter then.
        if (hundreds.PredictedValue == "0") return ("0", Sure(hundreds));

        int year = 5000 + int.Parse(hundreds.PredictedValue) * 100
                        + int.Parse(tens.PredictedValue) * 10
                        + int.Parse(units.PredictedValue);
        return (year.ToString(), Math.Min(Sure(hundreds), Math.Min(Sure(tens), Sure(units))));
    }

    /// <summary>
    /// Each digit model is tested on its own, but what matters is the whole year.
    /// Builds the year from all three on the test rows and checks it against the Year column.
    /// </summary>
    static void TestFullYear(MLContext ml, IDataView data,
        ITransformer hundredsModel, ITransformer tensModel, ITransformer unitsModel)
    {
        // Same split and seed as in LoadOrTrain, so these are the same unseen rows.
        var testSet = ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1).TestSet;

        Guess[] GuessAll(ITransformer model) =>
            ml.Data.CreateEnumerable<Guess>(model.Transform(testSet), reuseRowObject: false).ToArray();
        var hundreds = GuessAll(hundredsModel);
        var tens = GuessAll(tensModel);
        var units = GuessAll(unitsModel);
        var actual = testSet.GetColumn<string>("Year").ToArray();

        int correct = 0, sure = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            var (year, confidence) = CombineYear(hundreds[i], tens[i], units[i]);
            if (year == actual[i]) correct++;
            if (confidence >= MinConfidence) sure++;
        }

        Console.WriteLine($"Full year (all three digits together): " +
                          $"{correct / (double)actual.Length:P2} correct on unseen examples");
        Console.WriteLine($"    Sure enough for the server ({MinConfidence:P0}+): " +
                          $"{sure / (double)actual.Length:P1} of answers");
    }

    /// <summary>
    /// Loads a saved model, or trains and saves a new one. Either way, tests it.
    /// </summary>
    static ITransformer LoadOrTrain(MLContext ml, IDataView data, string column)
    {
        // Keep 20% of the examples aside, so we test on rows the model never saw.
        // Same seed every run, so a loaded model is tested on the same rows as a new one.
        var split = ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

        string path = Path.Combine(ModelsFolder, $"{column.ToLower()}.zip");
        if (File.Exists(path))
        {
            Console.WriteLine($"Loaded the {column} model from {path}, testing it...");
            var loaded = ml.Model.Load(path, out _);
            Test(ml, loaded, split.TestSet, column);
            return loaded;
        }

        var pipeline = ml.Transforms.Conversion.MapValueToKey("Label", column)
            // Turn the text into numbers: short letter sequences (e.g. "שבט", "״ו")
            // and words and word pairs, so the model sees both spelling and order.
            .Append(ml.Transforms.Text.FeaturizeText("Features", new TextFeaturizingEstimator.Options
            {
                CharFeatureExtractor = new WordBagEstimator.Options { NgramLength = 3, UseAllLengths = true },
                WordFeatureExtractor = new WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },
            }, nameof(DateRow.Text)))
            .Append(ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
            .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedValue", "PredictedLabel"));

        Console.WriteLine($"Training the {column} model... (this can take a few minutes)");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var model = pipeline.Fit(split.TrainSet);
        Console.WriteLine($"  Trained in {watch.Elapsed.TotalSeconds:N0} s");
        Test(ml, model, split.TestSet, column);

        Directory.CreateDirectory(ModelsFolder);
        ml.Model.Save(model, data.Schema, path);
        return model;
    }

    /// <summary>Prints how well a model does on the test rows.</summary>
    static void Test(MLContext ml, ITransformer model, IDataView testSet, string column)
    {
        var tested = model.Transform(testSet);
        var metrics = ml.MulticlassClassification.Evaluate(tested);
        Console.WriteLine($"  {column}: {metrics.MicroAccuracy:P2} correct on unseen examples");
        ReportWeakSpots(tested, column);
    }

    /// <summary>
    /// Two checks the overall accuracy can hide: the single value the model gets wrong most often,
    /// and how many answers are sure enough for the server to use.
    /// </summary>
    static void ReportWeakSpots(IDataView tested, string column)
    {
        var actual = tested.GetColumn<string>(column).ToArray();
        var predicted = tested.GetColumn<string>("PredictedValue").ToArray();
        var scores = tested.GetColumn<float[]>("Score").ToArray();

        var weakest = Enumerable.Range(0, actual.Length)
            .GroupBy(i => actual[i])
            .Select(g => (Value: g.Key, Correct: g.Count(i => predicted[i] == actual[i]) / (double)g.Count()))
            .MinBy(v => v.Correct);
        // Only for hundreds does 0 mean "no year"; for tens and units 0 is also a real digit.
        string weakestName = column == "YearHundreds" && weakest.Value == "0" ? "no year typed" : weakest.Value;
        Console.WriteLine($"    Weakest {column}: {weakestName}, {weakest.Correct:P1} correct");

        double sure = scores.Count(s => s.Length > 0 && s.Max() >= MinConfidence) / (double)scores.Length;
        Console.WriteLine($"    Sure enough for the server ({MinConfidence:P0}+): {sure:P1} of answers");
    }

    /// <summary>Lets you type dates and see what the models make of them.</summary>
    static void TryItYourself(MLContext ml, ITransformer day, ITransformer month,
        ITransformer hundreds, ITransformer tens, ITransformer units)
    {
        var dayEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(day);
        var monthEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(month);
        var hundredsEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(hundreds);
        var tensEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(tens);
        var unitsEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(units);

        Console.WriteLine();
        Console.WriteLine("Type a Hebrew date and press Enter (an empty line exits).");
        Console.WriteLine("Note: the terminal may show Hebrew letters in reverse order; that's only the display.");

        while (true)
        {
            Console.Write("> ");
            string? text = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(text)) break;

            var input = new DateRow { Text = text };
            var d = dayEngine.Predict(input);
            var m = monthEngine.Predict(input);
            var (year, yearSure) = CombineYear(
                hundredsEngine.Predict(input), tensEngine.Predict(input), unitsEngine.Predict(input));

            string yearText = year == "0" ? "not typed" : year;
            Console.WriteLine($"  Day: {d.PredictedValue} ({Sure(d):P0})  " +
                              $"Month: {m.PredictedValue} ({Sure(m):P0})  " +
                              $"Year: {yearText} ({yearSure:P0})");
            Console.WriteLine($"  Gregorian: {ToGregorian(d.PredictedValue, m.PredictedValue, year)}");
        }
    }

    /// <summary>How sure the model is of its own answer (0–100%).</summary>
    static float Sure(Guess g) => g.Score.Length == 0 ? 0 : g.Score.Max();

    /// <summary>Combines the three answers into a Gregorian date, if they make a real date.</summary>
    static string ToGregorian(string dayText, string monthLabel, string yearText)
    {
        if (yearText == "0") return "(no year typed)";
        if (!int.TryParse(dayText, out int day) || !int.TryParse(yearText, out int year))
            return "(could not read the answer)";

        var calendar = new HebrewCalendar();
        var months = calendar.IsLeapYear(year) ? LeapYearMonths : RegularYearMonths;
        int month = Array.IndexOf(months, monthLabel) + 1;
        if (month == 0) return $"({monthLabel} does not exist in {year})";

        try
        {
            return calendar.ToDateTime(year, month, day, 0, 0, 0, 0).ToString("dd/MM/yyyy");
        }
        catch (ArgumentOutOfRangeException)
        {
            return $"(day {day} does not exist in {monthLabel} {year})";
        }
    }
}
