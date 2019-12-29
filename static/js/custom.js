$(document).ready(function() {
  $('.en-naive').click(function (event){
    event.preventDefault();
    $.ajax({
      type: "GET",
      url: '../../templates/form_en_naive.html',
      success: function(response) {
        setTimeout(function(){
          $(".form-common").html(response)
        }, 2000)
      }
    });
  });
});