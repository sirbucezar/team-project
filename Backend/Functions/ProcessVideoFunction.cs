using System;
using System.IO;
using System.Net;
using System.Text.Json;
using System.Threading.Tasks;
using Azure.Storage.Queues;
using Azure.Storage.Queues.Models;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using backend.Models;
using backend.Helpers;

namespace backend.Functions
{
    public class ProcessVideoFunction
    {
        private readonly ILogger _logger;
        private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false
        };

        public ProcessVideoFunction(ILoggerFactory loggerFactory)
        {
            _logger = loggerFactory.CreateLogger<ProcessVideoFunction>();
        }

        [Function("ProcessVideo")]
        public async Task<HttpResponseData> RunProcessVideo(
            [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = "process_video")] HttpRequestData req)
        {
            _logger.LogInformation("ProcessVideo function triggered.");

            try
            {
                string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
                _logger.LogInformation($"Received request body: {requestBody}");

                if (string.IsNullOrWhiteSpace(requestBody))
                {
                    _logger.LogWarning("Empty request body received.");
                    return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.BadRequest, new { error = "Request body is required" });
                }

                ProcessVideoRequest videoRequest;
                try
                {
                    videoRequest = JsonSerializer.Deserialize<ProcessVideoRequest>(requestBody, JsonOptions);
                    if (videoRequest == null)
                    {
                        throw new JsonException("Deserialized request is null.");
                    }
                    _logger.LogInformation($"Processing request with ID: {videoRequest.ProcessingId}");
                }
                catch (JsonException jsonEx)
                {
                    _logger.LogError(jsonEx, "Invalid JSON format received.");
                    return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.BadRequest, new { error = "Invalid JSON format" });
                }

                if (string.IsNullOrWhiteSpace(videoRequest.ProcessingId))
                {
                    videoRequest.ProcessingId = Guid.NewGuid().ToString();
                    _logger.LogWarning($"Processing ID was missing, generated new: {videoRequest.ProcessingId}");
                }

                var storageConnString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
                var queueClient = new QueueClient(storageConnString, "videoprocess", new QueueClientOptions
                {
                    MessageEncoding = QueueMessageEncoding.Base64
                });
                await queueClient.CreateIfNotExistsAsync();

                var queueMessage = JsonSerializer.Serialize(videoRequest, JsonOptions);
                
                _logger.LogInformation($"Queue Message: {queueMessage}");
                await queueClient.SendMessageAsync(queueMessage);  // Automatic Base64 encoding

                var response = req.CreateResponse(HttpStatusCode.Accepted);
                await response.WriteAsJsonAsync(new { processing_id = videoRequest.ProcessingId, status = "accepted" });

                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing video request");
                return await ResponseHelpers.CreateJsonResponseAsync(req, HttpStatusCode.InternalServerError, new { error = "Internal server error" });
            }
        }
    }
}