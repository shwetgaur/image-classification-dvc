param(
    [string]$BucketName = "image-classification-dvc-746914061275",
    [string]$RoleName = "image-classification-dvc-ec2-role",
    [string]$ProfileName = "image-classification-dvc-ec2-profile",
    [string]$Region = "ap-south-1",
    [string]$InstanceId = ""
)

$trust = @'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
'@

$policy = @"
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::$BucketName",
      "arn:aws:s3:::$BucketName/*"
    ]
  }]
}
"@

aws iam create-role --role-name $RoleName --assume-role-policy-document $trust 2>$null
$PolicyName = "$RoleName-s3-read"
aws iam put-role-policy --role-name $RoleName --policy-name $PolicyName --policy-document $policy 2>$null
aws iam create-instance-profile --instance-profile-name $ProfileName 2>$null
aws iam add-role-to-instance-profile --instance-profile-name $ProfileName --role-name $RoleName 2>$null
Start-Sleep -Seconds 10

if (-not $InstanceId) {
    $InstanceId = Get-Content (Join-Path $PSScriptRoot "..\deploy\ec2_instance_id.txt") -ErrorAction SilentlyContinue
}
if (-not $InstanceId) {
    $InstanceId = aws ec2 describe-instances --filters "Name=tag:Name,Values=image-classification-dvc" "Name=instance-state-name,Values=running,pending" --query "Reservations[0].Instances[0].InstanceId" --output text --region $Region
}

aws ec2 associate-iam-instance-profile --instance-id $InstanceId --iam-instance-profile Name=$ProfileName --region $Region
Write-Host "Associated IAM profile $ProfileName with instance $InstanceId"
