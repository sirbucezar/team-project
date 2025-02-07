using System;
using System.Text.Json;
using System.Threading.Tasks;
using backend.Models;
using backend.Services;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;

namespace backend.Functions
{
    public class FeedbackAugmentationFunction
    {
        private readonly IAugmentationService _augmentationService;
        private readonly ILogger<FeedbackAugmentationFunction> _logger;

        public FeedbackAugmentationFunction(IAugmentationService augmentationService, ILoggerFactory loggerFactory)
        {
            _augmentationService = augmentationService;
            _logger = loggerFactory.CreateLogger<FeedbackAugmentationFunction>();
        }

        [Function("FeedbackAugmentationTrigger")]
        public async Task RunAsync([QueueTrigger("videoprocess-augmented", Connection = "AzureWebJobsStorage")] string queueMessage)
        {
            _logger.LogInformation($"Received augmentation queue message: {queueMessage}");

            try
            {
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                };
                var request = JsonSerializer.Deserialize<FeedbackAugmentationRequest>(queueMessage, options);
                if (request == null || string.IsNullOrEmpty(request.ProcessingId))
                {
                    _logger.LogError("Invalid or missing processing ID in queue message.");
                    return;
                }

                bool success = await _augmentationService.ProcessAugmentationAsync(request);

                if (success)
                {
                    _logger.LogInformation($"Augmentation completed successfully for Processing ID: {request.ProcessingId}");
                }
                else
                {
                    _logger.LogError($"Augmentation failed for Processing ID: {request.ProcessingId}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error processing augmentation queue message: {ex.Message}");
            }
        }
    }
}