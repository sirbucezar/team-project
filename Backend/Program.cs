using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Http;
using Polly;
using Polly.Extensions.Http;
using backend.Services;
using backend.Helpers;

var host = new HostBuilder()
    .ConfigureFunctionsWorkerDefaults()
    .ConfigureServices(services =>
    {
        // Register named HttpClient for Python service
        services.AddHttpClient("PythonServiceClient")
            .ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler
            {
                // Use only if you need to bypass certificate errors
                ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
            })
            .AddPolicyHandler(GetRetryPolicy());

        // Register custom services
        services.AddSingleton<ISasService, SasService>();
        services.AddSingleton<IStorageService, StorageService>();
        services.AddSingleton<IAugmentationService, AugmentationService>();

    })
    .ConfigureLogging(logging =>
    {
        logging.AddConsole();
    })
    .Build();

LogEnvironmentVariables();
host.Run();

static void LogEnvironmentVariables()
{
    var grpcPort = Environment.GetEnvironmentVariable("FUNCTIONS_GRPC_PORT") ?? "7071";
    var runtime = Environment.GetEnvironmentVariable("FUNCTIONS_WORKER_RUNTIME") ?? "dotnet-isolated";
    var port = Environment.GetEnvironmentVariable("WEBSITES_PORT") ?? "80";

    Environment.SetEnvironmentVariable("ASPNETCORE_URLS", $"http://*:{port}");

    Console.WriteLine($"[INFO] FUNCTIONS_GRPC_PORT: {grpcPort}");
    Console.WriteLine($"[INFO] FUNCTIONS_WORKER_RUNTIME: {runtime}");
}

static IAsyncPolicy<HttpResponseMessage> GetRetryPolicy()
{
    return HttpPolicyExtensions
        .HandleTransientHttpError()
        .WaitAndRetryAsync(3, retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)));
}