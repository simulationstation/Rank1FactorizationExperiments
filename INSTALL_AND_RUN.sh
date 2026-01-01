#!/bin/bash
set -e

echo "=== Installing Docker ==="
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

echo "=== Installing CVMFS ==="
wget -q https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb
sudo dpkg -i cvmfs-release-latest_all.deb
rm cvmfs-release-latest_all.deb
sudo apt-get update
sudo apt-get install -y cvmfs cvmfs-config-default

sudo bash -c 'cat > /etc/cvmfs/default.local << EOF
CVMFS_REPOSITORIES=cms.cern.ch
CVMFS_HTTP_PROXY=DIRECT
CVMFS_QUOTA_LIMIT=20000
EOF'

sudo cvmfs_config setup
sudo cvmfs_config probe cms.cern.ch

echo "=== Verifying ==="
ls /cvmfs/cms.cern.ch/cmsset_default.sh && echo "CVMFS: OK"
sudo docker run --rm hello-world && echo "Docker: OK"

echo ""
echo "=== INSTALLATION COMPLETE ==="
echo "Now run: newgrp docker"
echo "Then tell Claude to continue"
