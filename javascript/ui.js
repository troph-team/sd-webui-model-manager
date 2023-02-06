
function model_manager_download(){
    var id = randomId()
    requestProgress(id, gradioApp().getElementById('model_manager_download_panel'), null, function(){})

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}