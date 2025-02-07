using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using backend.Models;
using Microsoft.Extensions.Logging;

namespace backend.Services
{
    public class AugmentationService : IAugmentationService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<AugmentationService> _logger;
        private readonly string _pythonServiceUrl;

        private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false
        };

        public AugmentationService(HttpClient httpClient, ILoggerFactory loggerFactory)
        {
            _httpClient = httpClient;
            _logger = loggerFactory.CreateLogger<AugmentationService>();

            _pythonServiceUrl = Environment.GetEnvironmentVariable("PythonServiceUrl")
                ?? throw new InvalidOperationException("PythonServiceUrl is not configured.");
        }

        public async Task<bool> ProcessAugmentationAsync(FeedbackAugmentationRequest request)
        {
            _logger.LogInformation($"Starting augmentation for Processing ID: {request.ProcessingId}");

            try
            {
                var jsonContent = JsonSerializer.Serialize(request, JsonOptions);
                var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

                _logger.LogInformation($"Sending augmentation request to {_pythonServiceUrl}/augment");
                var response = await _httpClient.PostAsync($"{_pythonServiceUrl}/augment", content);

                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation($"Augmentation completed successfully for Processing ID: {request.ProcessingId}");
                    return true;
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Augmentation failed: {response.StatusCode} - {errorContent}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error calling augmentation API for Processing ID: {request.ProcessingId}. Error: {ex.Message}");
                return false;
            }
        }
    }
}