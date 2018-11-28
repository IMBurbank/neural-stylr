#!/bin/sh

##### CONSTANTS #####
DEFAULT_TAG='neural-stylr'


##### VARIABLES #####
tag_suffix='-gpu'


##### FLAGS #####
run_opts=			                # flag -r
runtime_opt="--runtime nvidia"      # flag -c (toggle use-cpu)
keep_opt="--rm"                     # flag -k (toggle keep container)
port="8888:8888"                    # flag -p
tag=""                              # flag -t
user="$(id -u):$(id -g)"            # flag -u
workdir="/workdir"                  # flag -w
volume="$(pwd):${workdir}:rw"       # flag -v

while getopts ":h:r:c:k:p:t:u:v:w:" flag; do
	case "${flag}" in
		h) 
			echo "HELP STUB"
			exit 0
			;;
		r) 
			run_opts="${OPTARG} ${run_opts}"
			;;
		c) 
			tag_suffix=
            runtime_opt=
			;;
		k)
			keep_opt=
			;;
		p) 
			port="${OPTARG}"
			;;
		t) 
			tag="${OPTARG}"
			;;
		u) 
			user=${OPTARG}
			;;
		v) 
			volume=${OPTARG}
			;;
		w) 
			workdir=${OPTARG}
			;;
		*) 
			echo "ERROR: Invalid flag ${flag}"
			exit 1
			;;
	esac
done
shift `expr $OPTIND - 1`


##### MAIN #####
run_opts="${run_opts} ${keep_opt} ${runtime_opt}"
[ -z "${tag}" ] && tag="${DEFAULT_TAG}${tag_suffix}"

if [ ! "$(docker ps -q -f name=${tag})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${tag})" ]; then
        # cleanup
		docker container rm "${tag}"
    fi
    # run container
	docker run \
		${run_opts} \
		--name "${tag}" \
		-p "${port}" \
		-u "${user}" \
		-w "${workdir}" \
		-v "${volume}" \
		-it \
		"${tag}" \
		"$@"
else
	# exec in running container
	docker exec -it "${tag}" bash -l "$@"
fi