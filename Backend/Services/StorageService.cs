using System;
using System.IO;
using System.Threading.Tasks;
using Azure.Storage.Blobs;

namespace backend.Services
{
    public class StorageService : IStorageService
    {
        private readonly string _connectionString;

        public StorageService()
        {
            _connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage") 
                ?? throw new InvalidOperationException("AzureWebJobsStorage is not configured");
        }

        public async Task CreateContainerIfNotExistsAsync(string containerName)
        {
            var blobServiceClient = new BlobServiceClient(_connectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            await containerClient.CreateIfNotExistsAsync();
        }

        public async Task<string> UploadBlobAsync(string containerName, string blobName, byte[] content)
        {
            var blobServiceClient = new BlobServiceClient(_connectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            await containerClient.CreateIfNotExistsAsync();
            var blobClient = containerClient.GetBlobClient(blobName);
            using var stream = new MemoryStream(content);
            await blobClient.UploadAsync(stream, overwrite: true);
            return blobClient.Uri.ToString();
        }
    }
}