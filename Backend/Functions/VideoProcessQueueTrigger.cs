using System;
using System.Text.Json;
using System.Threading.Tasks;
using Azure.Data.Tables;
using Azure.Storage.Blobs;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using backend.Models;
using System.Net.Http.Json;
using System.Collections.Generic;
using System.Net.Http;

namespace backend.Functions
{
    public class VideoProcessQueueTrigger
    {
        private readonly ILogger<VideoProcessQueueTrigger> _logger;
        private readonly HttpClient _httpClient;
        private readonly string _pythonServiceUrl;

        private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = null
        };

        public VideoProcessQueueTrigger(ILoggerFactory loggerFactory, IHttpClientFactory httpClientFactory)
        {
            _logger = loggerFactory.CreateLogger<VideoProcessQueueTrigger>();
            _httpClient = httpClientFactory.CreateClient("PythonServiceClient");
            _pythonServiceUrl = Environment.GetEnvironmentVariable("PythonServiceUrl")
                ?? throw new InvalidOperationException("PythonServiceUrl not configured");
        }

        [Function("VideoProcessQueueTrigger")]
        public async Task ProcessVideoQueue(
            [QueueTrigger("videoprocess", Connection = "AzureWebJobsStorage")] string queueMessage)
        {
            _logger.LogInformation("Received a message from the 'videoprocess' queue.");

            if (string.IsNullOrWhiteSpace(queueMessage))
            {
                _logger.LogError("Received an empty message from the queue.");
                throw new ArgumentException("Queue message is empty.");
            }

            try
            {
                _logger.LogInformation($"Encoded message received: {queueMessage}");

                // Since messages are now automatically Base64 encoded, no need to decode manually
                var message = JsonSerializer.Deserialize<VideoProcessMessage>(queueMessage, JsonOptions);

                if (message == null || string.IsNullOrWhiteSpace(message.ProcessingId))
                {
                    _logger.LogError("Deserialized message is invalid or missing 'processing_id'.");
                    throw new InvalidOperationException("Invalid message content or missing required fields.");
                }

                _logger.LogInformation($"Processing video request with ID: {message.ProcessingId}");

                await UpdateProcessingStatus(message.ProcessingId, "processing", "Video analysis started");

                var pythonRequest = new Dictionary<string, object>
                {
                    { "video_url", message.VideoUrl },
                    { "exercise", message.Exercise },
                    { "stages", message.Stages },
                    { "deployment_id", message.DeploymentId },
                    { "processing_id", message.ProcessingId }
                };

                _logger.LogInformation("Sending video processing request to the Python API.");
                var response = await _httpClient.PostAsJsonAsync($"{_pythonServiceUrl}/analyze", pythonRequest);

                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Python API returned error: {response.StatusCode} - {errorContent}");
                    throw new Exception($"Python service returned {response.StatusCode}: {errorContent}");
                }

                _logger.LogInformation($"Processing completed for ID: {message.ProcessingId}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing video queue message");
                throw;
            }
        }
        private async Task UpdateProcessingStatus(string processingId, string status, string message)
        {
            var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
            var tableClient = new TableClient(connectionString, "VideoProcessing");
            await tableClient.CreateIfNotExistsAsync();

            var entity = new TableEntity(processingId, "status")
            {
                { "Status", status },
                { "Message", message },
                { "LastUpdated", DateTimeOffset.UtcNow }
            };

            await tableClient.UpsertEntityAsync(entity);
            _logger.LogInformation($"Updated processing status for {processingId} to '{status}'.");
        }
    }
}