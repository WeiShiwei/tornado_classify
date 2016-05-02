$(function() {
	$("a").click(function(event){
		if($(this).attr("method") != "" && $(this).attr("method") != undefined)
		{
			event.preventDefault();
			$.ajax({
				url: $(this).attr("href"),
				type: $(this).attr("method"),
				complete: function(data){
					window.location.href = JSON.parse(data.responseText).url;
				}
			});
		}
	});
});
(function($){
  $.isBlank = function(obj){
    return(!obj || $.trim(obj) === "");
  };
})(jQuery);

Array.prototype.remove=function(dx)
{
    if(isNaN(dx)||dx>this.length){return false;}
    for(var i=0,n=0;i<this.length;i++)
    {
        if(this[i]!=this[dx])
        {
            this[n++]=this[i]
        }
    }
    this.length-=1
}