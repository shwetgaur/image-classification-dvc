param(
    [string]$Region = "ap-south-1",
    [string]$InstanceType = "t3.micro",
    [string]$KeyName = "image-classification-dvc-key"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$UserDataFile = Join-Path $PSScriptRoot "ec2_user_data.sh"
$UserData = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Content $UserDataFile -Raw)))

$VpcId = (aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $Region)
$SgName = "image-classification-dvc-sg"
$SgId = aws ec2 describe-security-groups --filters "Name=group-name,Values=$SgName" --query "SecurityGroups[0].GroupId" --output text --region $Region 2>$null
if (-not $SgId -or $SgId -eq "None") {
    $SgId = aws ec2 create-security-group --group-name $SgName --description "Image classification app" --vpc-id $VpcId --region $Region --query GroupId --output text
    aws ec2 authorize-security-group-ingress --group-id $SgId --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $Region | Out-Null
    aws ec2 authorize-security-group-ingress --group-id $SgId --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $Region | Out-Null
    aws ec2 authorize-security-group-ingress --group-id $SgId --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $Region | Out-Null
}

$Ami = aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023*" "Name=architecture,Values=x86_64" --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text --region $Region

$KeyPath = Join-Path (Join-Path $env:USERPROFILE ".ssh") "$KeyName.pem"
if (-not (Test-Path $KeyPath)) {
    aws ec2 create-key-pair --key-name $KeyName --query KeyMaterial --output text --region $Region | Out-File -Encoding ascii $KeyPath
}

$InstanceId = aws ec2 run-instances `
    --image-id $Ami `
    --instance-type $InstanceType `
    --key-name $KeyName `
    --security-group-ids $SgId `
    --user-data $UserData `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=image-classification-dvc}]" `
    --query "Instances[0].InstanceId" `
    --output text `
    --region $Region

Write-Host "InstanceId: $InstanceId"
aws ec2 wait instance-running --instance-ids $InstanceId --region $Region
$PublicIp = aws ec2 describe-instances --instance-ids $InstanceId --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region $Region
Write-Host "Public IP: $PublicIp"
Write-Host "App URL (after bootstrap ~5-10 min): http://$PublicIp/"
$PublicIp | Out-File (Join-Path $Root "deploy\ec2_public_ip.txt") -Encoding utf8
$InstanceId | Out-File (Join-Path $Root "deploy\ec2_instance_id.txt") -Encoding utf8
