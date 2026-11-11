# Step 1: build the server with the .NET SDK
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY server.csproj .
RUN dotnet restore
COPY . .
RUN dotnet publish -c Release -o /app

# Step 2: run it on the smaller runtime image (models and wwwroot come along)
FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .

# Render tells the server which port to use in $PORT; locally it falls back to 8080.
CMD ASPNETCORE_URLS=http://+:${PORT:-8080} dotnet server.dll
