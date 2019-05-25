# download Animals with Attributes dataset
declare -a URL=("https://cvml.ist.ac.at/AwA2/AwA2-base.zip" "https://cvml.ist.ac.at/AwA2/AwA2-data.zip")

for url in ${URL[@]}
do
	wget "$url" -P dataset/
done

for zipped in dataset/*.zip
do
	unzip "$zipped" -d dataset
done
