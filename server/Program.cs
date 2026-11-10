using System.Globalization;
using System.Text.Encodings.Web;
using System.Text.Unicode;
using Microsoft.Extensions.ML;

// Below this confidence we say "not sure" instead of guessing.
const float MinConfidence = 0.8f;

string ModelPath(string name) => Path.Combine(AppContext.BaseDirectory, "models", $"{name}.zip");

// The year comes from three models, one per letter-digit (תשפ״ז = 7, 8, 7).
foreach (var name in new[] { "day", "month", "yearhundreds", "yeartens", "yearunits" })
{
    if (!File.Exists(ModelPath(name)))
    {
        Console.WriteLine($"Model not found: {ModelPath(name)}");
        Console.WriteLine("Copy the models folder from train_AI into this folder, then run again.");
        return;
    }
}

var builder = WebApplication.CreateBuilder(args);

// Load the trained models once, when the server starts.
builder.Services.AddPredictionEnginePool<DateRow, Guess>()
    .FromFile("day", ModelPath("day"))
    .FromFile("month", ModelPath("month"))
    .FromFile("yearhundreds", ModelPath("yearhundreds"))
    .FromFile("yeartens", ModelPath("yeartens"))
    .FromFile("yearunits", ModelPath("yearunits"));

// Show Hebrew as Hebrew in the answers, not as \u05D8 codes.
builder.Services.ConfigureHttpJsonOptions(options =>
    options.SerializerOptions.Encoder = JavaScriptEncoder.Create(UnicodeRanges.All));

// Let web apps (like the Flutter web demo) call the server.
builder.Services.AddCors();

var app = builder.Build();

// The demo page: wwwroot/index.html is shown at the server's main address.
app.UseDefaultFiles();
app.UseStaticFiles();

app.UseCors(policy => policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());

app.MapGet("/parse", (string? text, PredictionEnginePool<DateRow, Guess> models) =>
{
    if (string.IsNullOrWhiteSpace(text))
        return Results.BadRequest(new { error = "Send the date as /parse?text=..." });

    var input = new DateRow { Text = text };
    var day = models.Predict("day", input);
    var month = models.Predict("month", input);
    var (year, yearSure) = HebrewDates.CombineYear(
        models.Predict("yearhundreds", input),
        models.Predict("yeartens", input),
        models.Predict("yearunits", input));

    var confidence = new Confidence(HebrewDates.Sure(day), HebrewDates.Sure(month), yearSure);
    var problems = new List<string>();

    // Keep each part only if its model is sure of it.
    int? dayNumber = confidence.Day >= MinConfidence ? int.Parse(day.PredictedValue) : null;
    string? monthName = confidence.Month >= MinConfidence ? month.PredictedValue : null;

    if (dayNumber is null && monthName is null) problems.Add("Could not find a day and a month");
    else if (dayNumber is null) problems.Add("Could not find a day");
    else if (monthName is null) problems.Add("Could not find a month");

    int? yearNumber = null;
    if (confidence.Year < MinConfidence) problems.Add("Could not read the year");
    else if (year == 0) problems.Add("No year was typed");
    else yearNumber = year;

    // A Gregorian date needs all three parts, and they must make a real date.
    string? gregorian = null;
    if (dayNumber is not null && monthName is not null && yearNumber is not null)
    {
        var (date, problem) = HebrewDates.ToGregorian(dayNumber.Value, monthName, yearNumber.Value);
        gregorian = date;
        if (problem is not null) problems.Add(problem);
    }

    return Results.Ok(new ParseResult(text, dayNumber, monthName, yearNumber,
        gregorian, problems, confidence));
});

app.Run();

/// <summary>The input the models expect: the same columns they were trained on.</summary>
public class DateRow
{
    public string Text { get; set; } = "";
    public string Day { get; set; } = "";
    public string Month { get; set; } = "";
    public string Year { get; set; } = "";
    public string YearHundreds { get; set; } = "";
    public string YearTens { get; set; } = "";
    public string YearUnits { get; set; } = "";
}

/// <summary>A model's answer and how sure it is of each option.</summary>
public class Guess
{
    public string PredictedValue { get; set; } = "";
    public float[] Score { get; set; } = [];
}

/// <summary>How sure each model is, from 0 to 1.</summary>
public record Confidence(float Day, float Month, float Year);

/// <summary>What the server answers.</summary>
public record ParseResult(
    string Text,
    int? Day,
    string? Month,
    int? Year,
    string? GregorianDate,
    List<string> Problems,
    Confidence Confidence);

static class HebrewDates
{
    static readonly string[] RegularYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "Adar",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];
    static readonly string[] LeapYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "AdarI", "AdarII",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];

    public static float Sure(Guess g) => g.Score.Length == 0 ? 0 : MathF.Round(g.Score.Max(), 2);

    /// <summary>
    /// Builds the year from the three digit answers: 7, 8, 7 → 5787. 0 means no year typed.
    /// The year is only as sure as its least sure digit. Same rule as in train_AI.
    /// </summary>
    public static (int Year, float Sure) CombineYear(Guess hundreds, Guess tens, Guess units)
    {
        // Hundreds 0 means no year was typed; the other two digits don't matter then.
        if (hundreds.PredictedValue == "0") return (0, Sure(hundreds));

        int year = 5000 + int.Parse(hundreds.PredictedValue) * 100
                        + int.Parse(tens.PredictedValue) * 10
                        + int.Parse(units.PredictedValue);
        return (year, Math.Min(Sure(hundreds), Math.Min(Sure(tens), Sure(units))));
    }

    /// <summary>Turns the answers into a Gregorian date (yyyy-MM-dd), or explains why it can't.</summary>
    public static (string? Date, string? Problem) ToGregorian(int day, string monthLabel, int year)
    {
        var calendar = new HebrewCalendar();

        // The calendar only knows years 5343–5999 (1583–2239); outside that it throws.
        int firstYear = calendar.GetYear(calendar.MinSupportedDateTime);
        int lastYear = calendar.GetYear(calendar.MaxSupportedDateTime);
        if (year < firstYear || year > lastYear)
            return (null, $"Year {year} is outside the supported range ({firstYear}–{lastYear})");

        var months = calendar.IsLeapYear(year) ? LeapYearMonths : RegularYearMonths;
        int month = Array.IndexOf(months, monthLabel) + 1;
        if (month == 0) return (null, $"{monthLabel} does not exist in {year}");

        try
        {
            var date = calendar.ToDateTime(year, month, day, 0, 0, 0, 0);
            return (date.ToString("yyyy-MM-dd"), null);
        }
        catch (ArgumentOutOfRangeException)
        {
            return (null, $"Day {day} does not exist in {monthLabel} {year}");
        }
    }
}
