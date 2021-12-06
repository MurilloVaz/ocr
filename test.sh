file=$1
if [ $# -eq 0 ]; then
    file="test_file.png"
fi
response=$(curl -s -L -F "file=@$file" http://127.0.0.1:8000/read/ -H 'Content-Type: multipart/form-data' -H "accept: */*" -X 'POST'  --insecure)
echo "Digito reconhecido: $response"
	 