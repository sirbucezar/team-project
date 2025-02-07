using System.Threading.Tasks;

namespace backend.Services
{
    public interface IStorageService
    {
        Task<string> UploadBlobAsync(string containerName, string blobName, byte[] content);
        Task CreateContainerIfNotExistsAsync(string containerName);
    }
}