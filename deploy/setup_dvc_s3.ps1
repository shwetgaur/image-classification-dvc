# Configure DVC remote on AWS S3 (Account: 746914061275, user: shwetgaur)
param(
    [string]$BucketName = "image-classification-dvc-746914061275",
    [string]$Region = "ap-south-1"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Creating S3 bucket (if not exists): $BucketName in $Region"
try {
    aws s3api create-bucket --bucket $BucketName --region $Region --create-bucket-configuration LocationConstraint=$Region 2>$null
} catch {
    aws s3api head-bucket --bucket $BucketName 2>$null
}

dvc remote add -d storage "s3://$BucketName/dvc-store" 2>$null
dvc remote modify storage region $Region

Write-Host "DVC remote configured:"
dvc remote list
Write-Host "Run: dvc push"
