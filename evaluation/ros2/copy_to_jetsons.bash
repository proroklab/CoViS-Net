for ip in 10.3.1.2 10.3.1.3; do
  rsync -a --progress --delete . nvidia@$ip:/home/nvidia/covisnet --exclude build --exclude log --exclude install --exclude logs #&
done
